// Phase 2: fine-tune forward_model_nn's MLP with the pairwise-ranking
// objective Sho Saga's writeup describes as the actual differentiator
// ("正解と同じ分子式の別の分子（おとり）をPubChemから集め、正解の点数が
// おとりより高くなるようにも学習させる" - forward-model-sho-saga.html).
// Phase 1 (train_forward_model_nn.cpp) only trains the MLP to match real
// peak intensities; that alone has no mechanism to learn "which real
// candidates fool it", which is the entire reason today's rule-based
// (untrained) version regressed the real LB despite a clean holdout number
// (see RESUME.md's v8 entry) - replacing the rule with a trained-but-only-
// regression-trained MLP would not by itself fix that gap. This phase
// starts from phase 1's weights and fine-tunes with a hinge loss between
// the true structure's score and same-(approximate-)formula decoys' scores
// against the same real spectrum, both scores computed the differentiable
// way (through the MLP, not by reading .data() out of the graph early).
//
// Decoys are the pool's own ppm-windowed same-neutral-mass candidates
// (forward_holdout.cpp's exact mechanism and the project's standing
// approximation for "same formula" throughout, since computing an exact
// formula key for millions of pool candidates up front isn't done here).
//
// Usage:
//   train_forward_model_rank <dir> <lib_stem> <in_mlp.bin> <max_molecules>
//                             <ppm> <max_decoys> <margin> <steps> <lr> <out.bin>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include <GraphMol/GraphMol.h>
#include <GraphMol/SmilesParse/SmilesParse.h>

#include "../autograd/optim.h"
#include "forward_model_nn.h"

namespace {

struct Head { float precursor; int32_t mol, off, n, tag; };

struct Store {
  std::vector<Head> head;
  std::vector<float> mz, inten;
  std::vector<std::string> name, smiles;
  const float* mzs(const Head& h) const { return mz.data() + h.off; }
  const float* its(const Head& h) const { return inten.data() + h.off; }
};

std::vector<char> slurp(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) { std::fprintf(stderr, "cannot read %s\n", path.c_str()); std::exit(1); }
  std::streamsize n = f.tellg();
  f.seekg(0);
  std::vector<char> buf((size_t(n)));
  f.read(buf.data(), n);
  return buf;
}

Store loadSpectra(const std::string& dir, const std::string& stem) {
  Store s;
  std::vector<char> hb = slurp(dir + "/" + stem + "_head.bin");
  s.head.resize(hb.size() / sizeof(Head));
  std::memcpy(s.head.data(), hb.data(), s.head.size() * sizeof(Head));
  std::vector<char> pb = slurp(dir + "/" + stem + "_peaks.bin");
  size_t total = 0;
  for (const Head& h : s.head) total += size_t(h.n);
  s.mz.resize(total); s.inten.resize(total);
  const float* p = reinterpret_cast<const float*>(pb.data());
  size_t cur = 0;
  for (const Head& h : s.head) {
    std::memcpy(s.mz.data() + cur, p, size_t(h.n) * sizeof(float)); p += h.n;
    std::memcpy(s.inten.data() + cur, p, size_t(h.n) * sizeof(float)); p += h.n;
    cur += size_t(h.n);
  }
  std::ifstream f(dir + "/" + stem + "_mols.txt");
  std::string line;
  while (std::getline(f, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    size_t t = line.find('\t');
    s.name.push_back(line.substr(0, t));
    s.smiles.push_back(t == std::string::npos ? "" : line.substr(t + 1));
  }
  std::printf("%s: %zu spectra, %zu molecules\n", stem.c_str(), s.head.size(), s.name.size());
  return s;
}

struct Pool {
  std::vector<float> mass;      // by mol_id
  std::vector<std::string> smiles;
};

Pool loadPool(const std::string& dir) {
  Pool p;
  std::ifstream f(dir + "/pool_head.bin", std::ios::binary | std::ios::ate);
  if (!f) { std::fprintf(stderr, "cannot read pool_head.bin\n"); std::exit(1); }
  size_t bytes = f.tellg(); f.seekg(0);
  int n = int(bytes / 12);
  std::vector<int32_t> molId(n);
  std::vector<float> massSorted(n);
  for (int i = 0; i < n; ++i) {
    int32_t dummy;
    f.read(reinterpret_cast<char*>(&massSorted[i]), 4);
    f.read(reinterpret_cast<char*>(&molId[i]), 4);
    f.read(reinterpret_cast<char*>(&dummy), 4);
  }
  std::ifstream mf(dir + "/pool_mols.txt");
  std::string line;
  while (std::getline(mf, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    size_t t = line.find('\t');
    p.smiles.push_back(t == std::string::npos ? "" : line.substr(t + 1));
  }
  p.mass.resize(p.smiles.size());
  for (int i = 0; i < n; ++i) p.mass[molId[i]] = massSorted[i];
  std::printf("pool: %zu candidates\n", p.smiles.size());
  return p;
}

constexpr float ADDUCT_SHIFT[10] = {
    1.007276f, -1.007276f, 22.989218f, 18.033823f, 38.963158f,
    34.969402f, 44.998201f, -17.002740f, 1.007276f, 22.989218f,
};

RDKit::RWMol* safeParse(const std::string& smi) {
  try { return RDKit::SmilesToMol(smi); } catch (...) { return nullptr; }
}

// Nearest real peak's intensity within tol of `mass`, or 0. mz is sorted.
float nearestIntensity(const float* mz, const float* it, int n, double mass, float tol) {
  int lo = 0, hi = n;
  while (lo < hi) {
    int mid = (lo + hi) / 2;
    if (double(mz[mid]) < mass - tol) lo = mid + 1; else hi = mid;
  }
  float best = 0.0f;
  for (int i = lo; i < n && double(mz[i]) <= mass + tol; ++i)
    best = std::max(best, it[i]);
  return best;
}

// Differentiable score for one candidate molecule against one real query
// spectrum: enumerate the candidate's own fragments, run the MLP (keeping
// everything as ag::Tensor so gradients flow), and dot the predicted
// intensities against the (fixed, non-grad) real-peak intensities at each
// fragment's own mass - normalised by the predicted side's own norm so the
// model can't win purely by inflating raw output magnitude. This is the
// unnormalised-numerator half of forward_holdout.cpp's cosine(); the
// query-side norm is omitted because it's the same constant for every
// candidate scored against a given query, so it cancels out of the hinge
// difference between true and decoy scores.
ag::Tensor differentiableScore(const RDKit::ROMol& mol, const forward_model_nn::MLP& mlp,
                                const forward_model_nn::Context& ctx,
                                const forward_model_nn::Params& fp,
                                const float* mz, const float* it, int n, float tol) {
  auto frags = forward_model_nn::enumerateFragments(mol);
  int nShift = 2 * fp.maxHShift + 1;
  int nRows = int(frags.size()) * nShift;
  std::vector<float> xs(size_t(nRows) * forward_model_nn::kNumFeatures);
  std::vector<float> targets(nRows);
  int row = 0;
  for (const auto& ff : frags)
    for (int k = -fp.maxHShift; k <= fp.maxHShift; ++k) {
      auto r = forward_model_nn::featureRow(ff, k, ctx);
      for (int c = 0; c < forward_model_nn::kNumFeatures; ++c)
        xs[size_t(row) * forward_model_nn::kNumFeatures + c] = r[c];
      double shifted = ff.mass + double(k) * fp.massH;
      targets[row] = nearestIntensity(mz, it, n, shifted, tol);
      ++row;
    }
  ag::Tensor x = ag::Tensor::from(xs, {nRows, forward_model_nn::kNumFeatures}, false);
  ag::Tensor y = mlp.forward(x);                                   // (nRows,1), differentiable
  ag::Tensor t = ag::Tensor::from(targets, {nRows, 1}, false);      // fixed
  ag::Tensor dot = ag::sum(ag::mul(y, t));                         // scalar
  ag::Tensor normPred = ag::sqrt_(ag::add_scalar(ag::sum(ag::mul(y, y)), 1e-6f));
  return ag::divide(dot, normPred);
}

}  // namespace

int main(int argc, char** argv) {
  std::string dir = argc > 1 ? argv[1] : ".";
  std::string libStem = argc > 2 ? argv[2] : "train";
  std::string inMlpPath = argc > 3 ? argv[3] : "fwd_mlp.bin";
  int maxMolecules = argc > 4 ? std::atoi(argv[4]) : 2000;
  float ppm = argc > 5 ? float(std::atof(argv[5])) : 20.0f;
  int maxDecoys = argc > 6 ? std::atoi(argv[6]) : 5;
  float margin = argc > 7 ? float(std::atof(argv[7])) : 0.1f;
  int steps = argc > 8 ? std::atoi(argv[8]) : 2000;
  float lr = argc > 9 ? float(std::atof(argv[9])) : 2e-4f;
  std::string outPath = argc > 10 ? argv[10] : "fwd_mlp_rank.bin";
  const float tol = 0.01f;
  forward_model_nn::Params fp;

  Store lib = loadSpectra(dir, libStem);
  Pool pool = loadPool(dir);
  std::vector<float> sortedMass = pool.mass;
  std::vector<int> order(pool.mass.size());
  for (size_t i = 0; i < order.size(); ++i) order[i] = int(i);
  std::sort(order.begin(), order.end(), [&](int a, int b) { return pool.mass[a] < pool.mass[b]; });
  for (size_t i = 0; i < order.size(); ++i) sortedMass[i] = pool.mass[order[i]];

  RDKit::PeriodicTable::getTable();
  forward_model_nn::MLP mlp = forward_model_nn::MLP::load(dir + "/" + inMlpPath);
  std::printf("loaded %s (inDim=%d hidden=%d)\n", inMlpPath.c_str(), mlp.inDim, mlp.hidden);
  ag::Adam opt(mlp.params(), lr, 0.9f, 0.999f, 1e-8f, 0.0f, true);

  std::unordered_map<int, std::vector<int>> byMol;
  for (size_t i = 0; i < lib.head.size(); ++i) byMol[lib.head[i].mol].push_back(int(i));
  std::vector<int> molIds;
  for (auto& kv : byMol) molIds.push_back(kv.first);
  std::sort(molIds.begin(), molIds.end());
  size_t stride = molIds.size() > size_t(maxMolecules) ? molIds.size() / size_t(maxMolecules) : 1;

  // Pre-select (molecule, one spectrum, decoy SMILES list) training examples
  // once, up front - decoy lookup (mass window + RDKit parse) is the
  // expensive part and doesn't depend on the model being trained.
  struct Example { int molId; int specIdx; std::vector<std::unique_ptr<RDKit::RWMol>> decoyMols; };
  std::vector<Example> examples;
  ag::seed(7);
  for (size_t mi = 0; mi < molIds.size() && examples.size() < size_t(maxMolecules); mi += stride) {
    int molId = molIds[mi];
    const std::string& smi = lib.smiles[molId];
    if (smi.empty()) continue;
    const auto& spectra = byMol[molId];
    if (spectra.empty()) continue;
    const Head& q = lib.head[spectra[0]];

    int adductIdx = q.tag < 0 ? -1 : q.tag / 16;
    float shift = (adductIdx >= 0 && adductIdx < 10) ? ADDUCT_SHIFT[adductIdx] : 1.007276f;
    float neutral = q.precursor - shift;
    float window = neutral * ppm * 1e-6f;
    size_t lo = std::lower_bound(sortedMass.begin(), sortedMass.end(), neutral - window) - sortedMass.begin();
    size_t hi = std::upper_bound(sortedMass.begin(), sortedMass.end(), neutral + window) - sortedMass.begin();
    if (hi <= lo) continue;

    std::vector<int> candIdx;
    for (size_t p = lo; p < hi; ++p) candIdx.push_back(order[p]);
    // shuffle and cap so a crowded mass window doesn't dominate cost
    for (size_t i = candIdx.size(); i > 1; --i) {
      size_t j = size_t(ag::randf() * float(i));
      std::swap(candIdx[i - 1], candIdx[j]);
    }
    Example ex; ex.molId = molId; ex.specIdx = spectra[0];
    for (size_t c = 0; c < candIdx.size() && int(ex.decoyMols.size()) < maxDecoys; ++c) {
      const std::string& dsmi = pool.smiles[candIdx[c]];
      if (dsmi.empty() || dsmi == smi) continue;
      std::unique_ptr<RDKit::RWMol> dm(safeParse(dsmi));
      if (!dm) continue;
      ex.decoyMols.push_back(std::move(dm));
    }
    if (ex.decoyMols.empty()) continue;
    examples.push_back(std::move(ex));
    if (examples.size() % 200 == 0)
      std::printf("  prepared %zu examples\n", examples.size());
  }
  std::printf("training set: %zu (molecule, decoys) examples, mean %.1f decoys/example\n",
              examples.size(),
              examples.empty() ? 0.0 : [&]{ size_t s=0; for (auto&e: examples) s+=e.decoyMols.size(); return double(s)/examples.size(); }());
  if (examples.empty()) { std::fprintf(stderr, "no training examples\n"); return 1; }

  for (int step = 0; step < steps; ++step) {
    const Example& ex = examples[size_t(step) % examples.size()];
    std::unique_ptr<RDKit::RWMol> trueMol(safeParse(lib.smiles[ex.molId]));
    if (!trueMol) continue;
    const Head& q = lib.head[ex.specIdx];
    const float* mz = lib.mzs(q); const float* it = lib.its(q);
    forward_model_nn::Context ctx;
    if (q.tag >= 0) { ctx.adductIdx = q.tag / 16; ctx.instrumentIdx = q.tag % 16; }

    float lrNow = ag::cosine_lr(step, steps, lr, std::min(50, steps / 10));
    opt.lr = lrNow;
    opt.zero_grad();

    ag::Tensor scoreTrue = differentiableScore(*trueMol, mlp, ctx, fp, mz, it, q.n, tol);
    ag::Tensor totalLoss;
    bool first = true;
    float lossVal = 0.0f;
    for (auto& decoyMol : ex.decoyMols) {
      ag::Tensor scoreDecoy = differentiableScore(*decoyMol, mlp, ctx, fp, mz, it, q.n, tol);
      // hinge(margin - (true - decoy)) = relu(margin - true + decoy)
      ag::Tensor diff = ag::add_scalar(ag::sub(scoreDecoy, scoreTrue), margin);
      ag::Tensor hinge = ag::relu(diff);
      totalLoss = first ? hinge : ag::add(totalLoss, hinge);
      first = false;
    }
    ag::Tensor loss = ag::mul_scalar(totalLoss, 1.0f / float(ex.decoyMols.size()));
    loss.backward();
    opt.step();
    lossVal = loss.item();

    if (step % 50 == 0) {
      std::printf("  step %4d  loss %.6f  lr %.6f  (%zu decoys)\n",
                  step, lossVal, lrNow, ex.decoyMols.size());
      std::fflush(stdout);
    }
  }

  mlp.save(dir + "/" + outPath);
  std::printf("wrote %s (inDim=%d hidden=%d)\n", outPath.c_str(), mlp.inDim, mlp.hidden);
  return 0;
}
