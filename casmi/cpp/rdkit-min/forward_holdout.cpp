// Does the forward model actually rank the true answer above same-mass
// decoys? The one question that matters before wiring this into the real
// pipeline or spending a submission on it (see the v7 analog-propagation
// regression - built on a GNN-free heuristic too, never locally validated,
// and it made the real score worse).
//
// For each held-out query molecule: find every pool candidate within a
// tight ppm window of its (adduct-corrected) neutral mass - these are its
// same-formula-ish decoys, the only competition a mass gate alone cannot
// resolve - predict each one's spectrum with the forward model, score
// against the real spectrum, and see where the true SMILES lands.
//
// Usage:
//   forward_holdout <dir> <qry_stem> <ppm> [max_queries] [h_shift_decay]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <GraphMol/GraphMol.h>
#include <GraphMol/SmilesParse/SmilesParse.h>

#include "forward_model.h"

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
  std::vector<float> mass;
  std::vector<std::string> smiles;
};

Pool loadPool(const std::string& dir) {
  Pool p;
  std::ifstream f(dir + "/pool_head.bin", std::ios::binary | std::ios::ate);
  if (!f) { std::fprintf(stderr, "cannot read pool_head.bin\n"); std::exit(1); }
  size_t bytes = f.tellg(); f.seekg(0);
  int n = int(bytes / 12);
  std::vector<int32_t> molId(n), hasSpec(n);
  p.mass.resize(n);
  for (int i = 0; i < n; ++i) {
    f.read(reinterpret_cast<char*>(&p.mass[i]), 4);
    f.read(reinterpret_cast<char*>(&molId[i]), 4);
    f.read(reinterpret_cast<char*>(&hasSpec[i]), 4);
  }
  std::vector<std::string> names;
  std::ifstream mf(dir + "/pool_mols.txt");
  std::string line;
  while (std::getline(mf, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    size_t t = line.find('\t');
    names.push_back(line.substr(0, t));
    p.smiles.push_back(t == std::string::npos ? "" : line.substr(t + 1));
  }
  // p.mass[i] / p.smiles indexing: mass array is mass-sorted by row, but
  // smiles/names are in original (mol_id) order - mol_id[i] tells us which.
  // Rebuild p.mass/p.smiles both in mol_id order so a single index works.
  std::vector<float> massByMol(names.size());
  for (int i = 0; i < n; ++i) massByMol[molId[i]] = p.mass[i];
  p.mass = massByMol;
  std::printf("pool: %zu candidates\n", p.smiles.size());
  return p;
}

constexpr float ADDUCT_SHIFT[10] = {
    1.007276f, -1.007276f, 22.989218f, 18.033823f, 38.963158f,
    34.969402f, 44.998201f, -17.002740f, 1.007276f, 22.989218f,
};

float cosine(const std::vector<std::pair<double, double>>& pred,
             const float* mz, const float* it, int n, float tol) {
  double dot = 0.0, na = 0.0, nb = 0.0;
  for (auto& pk : pred) na += pk.second;
  for (int i = 0; i < n; ++i) nb += double(it[i]);
  if (na <= 0.0 || nb <= 0.0) return 0.0f;
  for (auto& pk : pred) {
    double best = 0.0;
    for (int i = 0; i < n; ++i) {
      if (std::fabs(double(mz[i]) - pk.first) <= tol) best = std::max(best, double(it[i]));
    }
    if (best > 0.0) dot += std::sqrt(pk.second) * std::sqrt(best);
  }
  return float(dot / std::sqrt(na * nb + 1e-12));
}

}  // namespace

int main(int argc, char** argv) {
  std::string dir = argc > 1 ? argv[1] : ".";
  std::string qryStem = argc > 2 ? argv[2] : "qry";
  float ppm = argc > 3 ? float(std::atof(argv[3])) : 20.0f;
  int maxQueries = argc > 4 ? std::atoi(argv[4]) : 30;
  double hShiftDecay = argc > 5 ? std::atof(argv[5]) : 1.0;

  Store qry = loadSpectra(dir, qryStem);
  Pool pool = loadPool(dir);
  std::vector<float> sortedMass = pool.mass;
  std::vector<int> order(pool.mass.size());
  for (size_t i = 0; i < order.size(); ++i) order[i] = int(i);
  std::sort(order.begin(), order.end(), [&](int a, int b) { return pool.mass[a] < pool.mass[b]; });
  for (size_t i = 0; i < order.size(); ++i) sortedMass[i] = pool.mass[order[i]];

  forward_model::Params fp;
  fp.hShiftDecay = hShiftDecay;

  int nTested = 0, nFoundTop1 = 0, nFoundTop5 = 0, nHadDecoys = 0;
  double mrrSum = 0.0;

  for (size_t qi = 0; qi < qry.head.size() && nTested < maxQueries; ++qi) {
    const Head& q = qry.head[qi];
    const std::string& truthSmi = qry.smiles[q.mol];
    if (truthSmi.empty()) continue;

    int adductIdx = q.tag < 0 ? -1 : q.tag / 16;
    float shift = (adductIdx >= 0 && adductIdx < 10) ? ADDUCT_SHIFT[adductIdx] : 1.007276f;
    float neutral = q.precursor - shift;
    float window = neutral * ppm * 1e-6f;
    size_t lo = std::lower_bound(sortedMass.begin(), sortedMass.end(), neutral - window) - sortedMass.begin();
    size_t hi = std::upper_bound(sortedMass.begin(), sortedMass.end(), neutral + window) - sortedMass.begin();
    if (hi <= lo) continue;

    ++nTested;
    std::vector<std::pair<float, bool>> ranked;  // score, is_truth
    for (size_t p = lo; p < hi; ++p) {
      int idx = order[p];
      const std::string& smi = pool.smiles[idx];
      if (smi.empty()) continue;
      std::unique_ptr<RDKit::RWMol> mol(RDKit::SmilesToMol(smi));
      if (!mol) continue;
      auto pred = forward_model::predictSpectrum(*mol, fp);
      float s = cosine(pred, qry.mzs(q), qry.its(q), q.n, 0.01f);
      ranked.emplace_back(s, smi == truthSmi);
    }
    if (ranked.size() < 2) continue;  // no decoys, nothing to resolve
    ++nHadDecoys;
    std::sort(ranked.begin(), ranked.end(), [](auto& a, auto& b) { return a.first > b.first; });
    for (size_t r = 0; r < ranked.size(); ++r) {
      if (ranked[r].second) {
        mrrSum += 1.0 / double(r + 1);
        if (r == 0) ++nFoundTop1;
        if (r < 5) ++nFoundTop5;
        break;
      }
    }
    std::printf("  [%zu] candidates=%zu truth_rank=%s\n", qi, ranked.size(),
                [&]() {
                  for (size_t r = 0; r < ranked.size(); ++r)
                    if (ranked[r].second) return std::to_string(r + 1);
                  return std::string("not-found");
                }().c_str());
  }

  std::printf("\ntested %d queries with >=1 decoy (of %d with any candidates)\n", nHadDecoys, nTested);
  if (nHadDecoys > 0) {
    std::printf("top1 %.1f%%  top5 %.1f%%  MRR %.3f\n",
                100.0 * nFoundTop1 / nHadDecoys, 100.0 * nFoundTop5 / nHadDecoys,
                mrrSum / nHadDecoys);
  }
  return 0;
}
