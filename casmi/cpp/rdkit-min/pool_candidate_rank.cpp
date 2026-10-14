// Generate NEW candidates directly from the structure-only pool
// (pool_head.bin/pool_mols.txt, now 4.37M after the PubChem expansion),
// scored against each query's real spectrum by the trained MLP
// (forward_model_nn) - unlike forward_rerank_nn.cpp, this does not touch
// match.cpp's own library-search candidates at all, it is the OTHER half
// of the pipeline match.cpp structurally cannot do: a molecule whose true
// structure has no reference spectrum anywhere in train (the ~84% Sho
// Saga's writeup and this project's own denovo holdout both measure)
// can never appear in match.cpp's output no matter how the search is
// tuned, because match.cpp only ever returns train's own structures.
//
// For each query molecule: gather every pool candidate within a ppm
// window of its own (adduct-corrected) neutral mass, score each with the
// trained MLP against the best-matching of the molecule's own spectra,
// and keep the top K.
//
// Usage:
//   pool_candidate_rank <dir> <qry_stem> <mlp.bin> <ppm> <topk> <out.csv> [pool_stem] [names] [intensity] [score]
// pool_stem defaults to "pool" (<dir>/<pool_stem>_head.bin, _mols.txt);
// "names" writes the pool's InChIKey14 instead of SMILES, for evaluation.
// [intensity] "mlp" (default) or "uniform"; [score] "cos" (default) or eps
// for the background-mixture log-likelihood (loglik()).
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <GraphMol/GraphMol.h>
#include <GraphMol/SmilesParse/SmilesParse.h>

#include "forward_model_nn.h"

namespace {

struct Head { float precursor; int32_t mol, off, n, tag; };

struct Store {
  std::vector<Head> head;
  std::vector<float> mz, inten;
  std::vector<std::string> name;
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
  }
  std::printf("%s: %zu spectra, %zu molecules\n", stem.c_str(), s.head.size(), s.name.size());
  return s;
}

struct Pool {
  std::vector<float> mass;
  std::vector<std::string> smiles;
  std::vector<std::string> name;
};

Pool loadPool(const std::string& dir, const std::string& stem) {
  Pool p;
  std::ifstream f(dir + "/" + stem + "_head.bin", std::ios::binary | std::ios::ate);
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
  std::ifstream mf(dir + "/" + stem + "_mols.txt");
  std::string line;
  while (std::getline(mf, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    size_t t = line.find('\t');
    p.name.push_back(line.substr(0, t));
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

float cosine(const std::vector<std::pair<double, double>>& pred,
             const float* mz, const float* it, int n, float tol) {
  double dot = 0.0, na = 0.0, nb = 0.0;
  for (auto& pk : pred) na += pk.second;
  for (int i = 0; i < n; ++i) nb += double(it[i]);
  if (na <= 0.0 || nb <= 0.0) return 0.0f;
  for (auto& pk : pred) {
    double best = 0.0;
    for (int i = 0; i < n; ++i)
      if (std::fabs(double(mz[i]) - pk.first) <= tol) best = std::max(best, double(it[i]));
    if (best > 0.0) dot += std::sqrt(pk.second) * std::sqrt(best);
  }
  return float(dot / std::sqrt(na * nb + 1e-12));
}

// Normalized log-likelihood of the observed peaks under the candidate's
// predicted spectrum, as a mixture with a flat background:
//   q(m) = (1-eps) * sum_{|m_f - m| <= tol} I_f / sum I + eps * 2tol / range
//   score = sum_j w_j log q(m_j),  w_j = sqrt(I_j) / sum sqrt(I)
// Unlike the cosine, every observed peak the candidate cannot produce costs
// log(eps * 2tol/range) explicitly, and a candidate that spreads intensity
// over many fragments pays for it through the normalization - the bias the
// cosine has toward fragment-rich (flexible, many-bridge) structures.
float loglik(const std::vector<std::pair<double, double>>& pred,
             const float* mz, const float* it, int n, float tol, float eps, float range) {
  double tot = 0.0, wsum = 0.0;
  for (auto& pk : pred) tot += pk.second;
  for (int i = 0; i < n; ++i) wsum += std::sqrt(std::max(0.0, double(it[i])));
  if (tot <= 0.0 || wsum <= 0.0) return -1e9f;
  double bg = eps * 2.0 * tol / std::max(1.0f, range);
  double ll = 0.0;
  for (int i = 0; i < n; ++i) {
    double m = 0.0;
    for (auto& pk : pred)
      if (std::fabs(double(mz[i]) - pk.first) <= tol) m += pk.second;
    double w = std::sqrt(std::max(0.0, double(it[i]))) / wsum;
    ll += w * std::log((1.0 - eps) * m / tot + bg);
  }
  return float(ll);
}

}  // namespace

int main(int argc, char** argv) {
  std::string dir = argc > 1 ? argv[1] : ".";
  std::string qryStem = argc > 2 ? argv[2] : "test";
  std::string mlpPath = argc > 3 ? argv[3] : "fwd_mlp_rank_v2.bin";
  float ppm = argc > 4 ? float(std::atof(argv[4])) : 5.0f;
  int topk = argc > 5 ? std::atoi(argv[5]) : 25;
  std::string outPath = argc > 6 ? argv[6] : "pool_candidates.csv";
  std::string poolStem = argc > 7 ? argv[7] : "pool";
  bool writeNames = argc > 8 && std::string(argv[8]) == "names";
  // Intensity model: "mlp" (default) or "uniform" (every enumerated fragment
  // at equal intensity - the diagnostic for whether the learned intensities
  // carry any information on a given chemistry).
  std::string intensityMode = argc > 9 ? argv[9] : "mlp";
  // Peak score: "cos" (default; best over the molecule's spectra) or a
  // number eps > 0 for loglik() with that background weight (mean over spectra).
  std::string scoreMode = argc > 10 ? argv[10] : "cos";
  bool useLik = scoreMode != "cos";
  float eps = useLik ? float(std::atof(scoreMode.c_str())) : 0.0f;

  Store qry = loadSpectra(dir, qryStem);
  Pool pool = loadPool(dir, poolStem);
  std::vector<float> sortedMass = pool.mass;
  std::vector<int> order(pool.mass.size());
  for (size_t i = 0; i < order.size(); ++i) order[i] = int(i);
  std::sort(order.begin(), order.end(), [&](int a, int b) { return pool.mass[a] < pool.mass[b]; });
  for (size_t i = 0; i < order.size(); ++i) sortedMass[i] = pool.mass[order[i]];

  RDKit::PeriodicTable::getTable();
  forward_model_nn::MLP mlp = forward_model_nn::MLP::load(dir + "/" + mlpPath);
  std::printf("loaded %s (inDim=%d hidden=%d)\n", mlpPath.c_str(), mlp.inDim, mlp.hidden);

  std::unordered_map<int, std::vector<int>> byMol;
  for (size_t i = 0; i < qry.head.size(); ++i) byMol[qry.head[i].mol].push_back(int(i));
  std::vector<int> molIds;
  for (auto& kv : byMol) molIds.push_back(kv.first);
  std::sort(molIds.begin(), molIds.end());

  std::vector<std::string> outRows(molIds.size());
  std::atomic<size_t> next(0);
  std::mutex printMutex;

  unsigned nthread = std::max(1u, std::thread::hardware_concurrency());
  std::vector<std::thread> pool_threads;
  for (unsigned t = 0; t < nthread; ++t) {
    pool_threads.emplace_back([&]() {
      for (;;) {
        size_t mi = next++;
        if (mi >= molIds.size()) break;
        int molId = molIds[mi];

        std::set<int> candIdx;
        for (int qi : byMol[molId]) {
          const Head& q = qry.head[qi];
          int adductIdx = q.tag < 0 ? -1 : (q.tag % 16 == 15 ? q.tag / 16 + 1 : q.tag / 16);
          float shift = (adductIdx >= 0 && adductIdx < 10) ? ADDUCT_SHIFT[adductIdx] : 1.007276f;
          float neutral = q.precursor - shift;
          if (adductIdx == 8 || adductIdx == 9) neutral *= 0.5f;  // [2M+H]+, [2M+Na]+: a dimer
          float window = neutral * ppm * 1e-6f;
          size_t lo = std::lower_bound(sortedMass.begin(), sortedMass.end(), neutral - window) - sortedMass.begin();
          size_t hi = std::upper_bound(sortedMass.begin(), sortedMass.end(), neutral + window) - sortedMass.begin();
          for (size_t p = lo; p < hi; ++p) candIdx.insert(order[p]);
        }

        std::vector<std::pair<float, int>> ranked;
        for (int idx : candIdx) {
          const std::string& smi = pool.smiles[idx];
          if (smi.empty()) continue;
          std::unique_ptr<RDKit::RWMol> mol(safeParse(smi));
          if (!mol) continue;
          auto frags = forward_model_nn::enumerateFragments(*mol);
          float best = 0.0f;
          int nsp = 0;
          for (int qi : byMol[molId]) {
            const Head& q = qry.head[qi];
            forward_model_nn::Context ctx;
            if (q.tag >= 0) {  // see match.cpp adduct(): instrument 15 = unknown-instrument encoding
              bool unkInstr = q.tag % 16 == 15;
              int a = unkInstr ? q.tag / 16 + 1 : q.tag / 16;
              ctx.adductIdx = a < forward_model_nn::Context::kNumAdducts ? a : -1;
              ctx.instrumentIdx = unkInstr ? -1 : q.tag % 16;
            }
            auto pred = forward_model_nn::scoreFragmentsWithContext(frags, mlp, ctx);
            if (intensityMode == "uniform")
              for (auto& pk : pred) pk.second = 1.0;
            if (useLik) {
              best += loglik(pred, qry.mzs(q), qry.its(q), q.n, 0.01f, eps, q.precursor);
              ++nsp;
            } else {
              best = std::max(best, cosine(pred, qry.mzs(q), qry.its(q), q.n, 0.01f));
            }
          }
          if (useLik && nsp > 0) best /= float(nsp);
          ranked.emplace_back(best, idx);
        }
        std::sort(ranked.begin(), ranked.end(), [](auto& a, auto& b) { return a.first > b.first; });

        std::string cell;
        int placed = 0;
        for (auto& pr : ranked) {
          if (placed >= topk) break;
          if (!cell.empty()) cell += ';';
          cell += writeNames ? pool.name[pr.second] : pool.smiles[pr.second];
          ++placed;
        }
        outRows[mi] = qry.name[molId] + "," + cell;

        if (mi % 50 == 0) {
          std::lock_guard<std::mutex> lk(printMutex);
          std::printf("  [%s] %zu pool candidates -> top%d\n", qry.name[molId].c_str(),
                      candIdx.size(), placed);
        }
      }
    });
  }
  for (auto& th : pool_threads) th.join();

  std::ofstream out(dir + "/" + outPath);
  out << "molecule_id,smiles\n";
  for (auto& r : outRows) out << r << "\n";
  std::printf("\nwrote %s for %zu molecules\n", outPath.c_str(), molIds.size());
  return 0;
}
