// Train forward_model_nn's MLP to predict each enumerated fragment's
// relative intensity from real training spectra, instead of hand-tuning the
// exp(-decay*k^2) rule against one holdout. This is Sho Saga's writeup's
// primary training target ("予想スペクトルが実測に近づくように学習") -
// the secondary, decoy-ranking objective ("正解がおとりより高い点数になる
// ように学習") is a planned next step once this baseline is itself
// validated on a holdout, not assumed to help without measuring it (see
// RESUME.md's v8 regression: a holdout-validated change made the real LB
// worse once already today).
//
// For each selected training molecule: enumerate fragments once (shared
// across all of that molecule's spectra - the structure doesn't change),
// then for each spectrum, each (fragment, H-shift) row's target is the
// nearest real peak's intensity within tol, relative to that spectrum's own
// base peak (0 if nothing real lands there). mz arrays are sorted per
// spectrum (export_bin.py's own invariant, which match.cpp's cosine()
// already assumes), so the lookup is a binary search, not a scan.
//
// Usage:
//   train_forward_model_nn <dir> <lib_stem> <max_molecules> <hidden>
//                           <steps> <batch> <lr> <out.bin>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
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
  std::vector<char> buf((size_t(n)));  // extra parens: avoid most-vexing-parse
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

}  // namespace

int main(int argc, char** argv) {
  std::string dir = argc > 1 ? argv[1] : ".";
  std::string libStem = argc > 2 ? argv[2] : "train";
  int maxMolecules = argc > 3 ? std::atoi(argv[3]) : 5000;
  int hidden = argc > 4 ? std::atoi(argv[4]) : 16;
  int steps = argc > 5 ? std::atoi(argv[5]) : 2000;
  int batch = argc > 6 ? std::atoi(argv[6]) : 256;
  float lr = argc > 7 ? float(std::atof(argv[7])) : 1e-3f;
  std::string outPath = argc > 8 ? argv[8] : "fwd_mlp.bin";
  // Real spectra are mass-sparse: the overwhelming majority of enumerated
  // (fragment, shift) rows land where no real peak exists (target 0). An
  // unweighted MSE's global optimum is then "predict ~0 everywhere" - loss
  // goes to ~0 but the model has no discriminating power left (measured:
  // holdout MRR 0.151, worse than the un-trained fixed rule's 0.538 on the
  // same pool/ppm - see RESUME.md). negWeight downweights those zero rows
  // in the loss instead of dropping them, so the rare real-peak rows (the
  // only ones that matter for telling isomers apart) aren't drowned out.
  float negWeight = argc > 9 ? float(std::atof(argv[9])) : 0.05f;
  const float tol = 0.01f;
  forward_model_nn::Params fp;

  Store lib = loadSpectra(dir, libStem);
  RDKit::PeriodicTable::getTable();

  std::unordered_map<int, std::vector<int>> byMol;
  for (size_t i = 0; i < lib.head.size(); ++i) byMol[lib.head[i].mol].push_back(int(i));
  std::vector<int> molIds;
  for (auto& kv : byMol) molIds.push_back(kv.first);
  std::sort(molIds.begin(), molIds.end());
  // Stride-sample across the whole id range instead of taking the first N,
  // so the training set isn't biased toward however train_mols.txt happens
  // to be ordered.
  size_t stride = molIds.size() > size_t(maxMolecules)
      ? molIds.size() / size_t(maxMolecules) : 1;

  std::vector<float> X;
  std::vector<float> Y;
  int nMolUsed = 0, nSpectraUsed = 0;
  for (size_t mi = 0; mi < molIds.size() && nMolUsed < maxMolecules; mi += stride) {
    int molId = molIds[mi];
    const std::string& smi = lib.smiles[molId];
    if (smi.empty()) continue;
    std::unique_ptr<RDKit::RWMol> mol(safeParse(smi));
    if (!mol) continue;
    auto frags = forward_model_nn::enumerateFragments(*mol);
    ++nMolUsed;

    const auto& spectra = byMol[molId];
    int capSpectraPerMol = 3;  // avoid a handful of heavily-resampled
                                // molecules dominating the training set
    for (int si = 0; si < int(spectra.size()) && si < capSpectraPerMol; ++si) {
      const Head& q = lib.head[spectra[si]];
      if (q.n <= 0) continue;
      float base = 0.0f;
      for (int i = 0; i < q.n; ++i) base = std::max(base, lib.its(q)[i]);
      if (base <= 0.0f) continue;
      ++nSpectraUsed;
      forward_model_nn::Context ctx;
      if (q.tag >= 0) { ctx.adductIdx = q.tag / 16; ctx.instrumentIdx = q.tag % 16; }
      for (const auto& ff : frags) {
        for (int k = -fp.maxHShift; k <= fp.maxHShift; ++k) {
          auto row = forward_model_nn::featureRow(ff, k, ctx);
          for (float v : row) X.push_back(v);
          double shifted = ff.mass + double(k) * fp.massH;
          float target = nearestIntensity(lib.mzs(q), lib.its(q), q.n, shifted, tol) / base;
          Y.push_back(target);
        }
      }
    }
    if (nMolUsed % 500 == 0)
      std::printf("  %d molecules, %d spectra, %zu rows\n", nMolUsed, nSpectraUsed, Y.size());
  }
  std::printf("training set: %d molecules, %d spectra, %zu (fragment,shift) rows\n",
              nMolUsed, nSpectraUsed, Y.size());
  if (Y.empty()) { std::fprintf(stderr, "no training rows\n"); return 1; }

  forward_model_nn::MLP mlp = forward_model_nn::MLP::init(forward_model_nn::kNumFeatures, hidden);
  ag::Adam opt(mlp.params(), lr, 0.9f, 0.999f, 1e-8f, 0.0f, true);

  std::vector<int> order(Y.size());
  for (size_t i = 0; i < order.size(); ++i) order[i] = int(i);
  ag::seed(123);
  auto shuffle = [&]() {
    for (size_t i = order.size(); i > 1; --i) {
      size_t j = size_t(ag::randf() * float(i));
      std::swap(order[i - 1], order[j]);
    }
  };

  size_t cursor = order.size();
  for (int step = 0; step < steps; ++step) {
    if (cursor + size_t(batch) > order.size()) { shuffle(); cursor = 0; }
    int n = std::min(size_t(batch), order.size() - cursor);
    std::vector<float> xb(size_t(n) * forward_model_nn::kNumFeatures), yb(n), sqrtWb(n);
    float wSum = 0.0f;
    for (int i = 0; i < n; ++i) {
      int r = order[cursor + i];
      std::memcpy(&xb[size_t(i) * forward_model_nn::kNumFeatures],
                  &X[size_t(r) * forward_model_nn::kNumFeatures],
                  forward_model_nn::kNumFeatures * sizeof(float));
      yb[i] = Y[r];
      float w = yb[i] > 1e-6f ? 1.0f : negWeight;
      sqrtWb[i] = std::sqrt(w);
      wSum += w;
    }
    cursor += n;

    float lrNow = ag::cosine_lr(step, steps, lr, std::min(50, steps / 10));
    opt.lr = lrNow;
    opt.zero_grad();
    ag::Tensor x = ag::Tensor::from(xb, {n, forward_model_nn::kNumFeatures}, false);
    ag::Tensor target = ag::Tensor::from(yb, {n, 1}, false);
    ag::Tensor sqrtW = ag::Tensor::from(sqrtWb, {n, 1}, false);
    ag::Tensor pred = mlp.forward(x);
    // Weighted MSE via scale-then-square: (sqrt(w)*diff)^2 == w*diff^2, so
    // this reuses mul/sum rather than needing a dedicated weighted-mean op.
    ag::Tensor diff = ag::mul(ag::sub(pred, target), sqrtW);
    ag::Tensor loss = ag::mul_scalar(ag::sum(ag::mul(diff, diff)), 1.0f / wSum);
    loss.backward();
    opt.step();

    if (step % 100 == 0) {
      std::printf("  step %4d  loss %.6f  lr %.5f\n", step, loss.item(), lrNow);
      std::fflush(stdout);
    }
  }

  mlp.save(dir + "/" + outPath);
  std::printf("wrote %s (inDim=%d hidden=%d)\n", outPath.c_str(), mlp.inDim, mlp.hidden);
  return 0;
}
