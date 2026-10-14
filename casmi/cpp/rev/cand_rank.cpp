// Candidate retrieval + ranking for molecules the spectral library cannot
// answer, against WHOLE PubChem at runtime (pc_build.cpp's mass-sorted
// table) plus the precomputed pool (train + COCONUT). The visible
// test.parquet is a placeholder; the scoring rerun swaps in a hidden test,
// so nothing here may be keyed to the visible molecules - every candidate
// comes from a +-ppm window around whatever precursor arrives.
//
// Score = reverse-model fingerprint log-likelihood (sum of logit(p) over the
// candidate's ECFP on-bits, rev_model.hpp) + popWeight * popularity, where
// popularity = log1p(#PubChem substances) + log1p(#PubMed refs) (pop_build).
// PubChem candidates are parsed with rdkit-min and fingerprinted here; pool
// candidates use pool_ecfp.bin. Duplicates (same ECFP - stereo-blind, as is
// the metric's InChIKey14) are merged, keeping the best score.
//
//   cand_rank rank <dir> <qry> <models> <ppm> <topk> <out.csv> <popW> [skipTrain] [poolStem] [usePubChem]
//   cand_rank eval <dir> <qry> <models> <ppm> <popW> [skipTrain]
// skipTrain=1 drops pool entries that are train structures (has_spectrum)
// - on the denovo holdout those are where the answer would trivially be;
// without them the eval measures whether PubChem/COCONUT actually reach it.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <GraphMol/GraphMol.h>
#include <GraphMol/PeriodicTable.h>
#include <GraphMol/SmilesParse/SmilesParse.h>

#include "ecfp.h"
#include "rev_model.hpp"

using namespace rev;

namespace {

std::vector<char> slurp(const std::string& path, bool required = true) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    if (required) { std::fprintf(stderr, "cannot read %s\n", path.c_str()); std::exit(1); }
    return {};
  }
  std::streamsize n = f.tellg();
  f.seekg(0);
  std::vector<char> buf((size_t(n)));
  f.read(buf.data(), n);
  return buf;
}

// The exported tag is adduct*16 + instrument with -1 for unknowns, which
// silently corrupts when only the instrument is unknown: adduct a with
// instrument -1 gives 16a-1, i.e. adduct a-1 / instrument 15. Instrument 15
// can arise no other way (only 5 instruments exist), so it identifies the
// case and is decoded back here - no re-export needed. ~10% of train's
// spectra use instrument names outside the 5-name vocabulary.
inline void decodeTag(int32_t tag, int& adduct, int& instr) {
  if (tag < 0) { adduct = -1; instr = (tag >= -16) ? tag + 16 : -1; return; }
  if (tag % 16 == 15) { adduct = tag / 16 + 1; instr = -1; return; }
  adduct = tag / 16; instr = tag % 16;
}

struct Store {
  std::vector<Head> head;
  std::vector<float> mz, inten;
  std::vector<std::string> name;
};

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
  for (Head& h : s.head) {
    std::memcpy(s.mz.data() + cur, p, size_t(h.n) * 4); p += h.n;
    std::memcpy(s.inten.data() + cur, p, size_t(h.n) * 4); p += h.n;
    h.off = int32_t(cur);
    cur += size_t(h.n);
    int a, i;
    decodeTag(h.tag, a, i);
    h.tag = (a >= 0 ? a : 15) * 16 + (i >= 0 ? i : 15);  // canonical: 15 = unknown
  }
  std::ifstream f(dir + "/" + stem + "_mols.txt");
  std::string line;
  while (std::getline(f, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    s.name.push_back(line.substr(0, line.find('\t')));
  }
  std::printf("%s: %zu spectra, %zu molecules\n", stem.c_str(), s.head.size(), s.name.size());
  return s;
}

struct Fps { std::vector<uint32_t> ptr; std::vector<uint16_t> bits; };
Fps loadFps(const std::string& path) {
  Fps f;
  std::vector<char> b = slurp(path);
  const char* p = b.data();
  uint32_t n, nb; std::memcpy(&n, p, 4); std::memcpy(&nb, p + 4, 4); p += 8;
  f.ptr.resize(n + 1); f.ptr[0] = 0;
  for (uint32_t i = 0; i < n; ++i) {
    uint16_t c; std::memcpy(&c, p, 2); p += 2;
    size_t old = f.bits.size();
    f.bits.resize(old + c);
    std::memcpy(f.bits.data() + old, p, 2 * size_t(c)); p += 2 * size_t(c);
    f.ptr[i + 1] = uint32_t(f.bits.size());
  }
  return f;
}

uint64_t hashBits(const uint16_t* b, size_t n) {
  uint64_t h = 1469598103934665603ull;
  for (size_t k = 0; k < n; ++k) h = (h ^ b[k]) * 1099511628211ull;
  return h;
}

struct Pool {
  std::vector<float> sortedMass;
  std::vector<int32_t> order;
  std::vector<int32_t> hasSpec;   // by mol id
  std::vector<std::string> smiles;
  Fps fps;
};
Pool loadPool(const std::string& dir, const std::string& stem) {
  Pool p;
  std::vector<char> hb = slurp(dir + "/" + stem + "_head.bin");
  size_t n = hb.size() / 12;
  p.sortedMass.resize(n); p.order.resize(n); p.hasSpec.assign(n, 0);
  for (size_t i = 0; i < n; ++i) {
    int32_t mol, hs;
    std::memcpy(&p.sortedMass[i], hb.data() + 12 * i, 4);
    std::memcpy(&mol, hb.data() + 12 * i + 4, 4);
    std::memcpy(&hs, hb.data() + 12 * i + 8, 4);
    p.order[i] = mol;
    if (mol >= 0 && size_t(mol) < n) p.hasSpec[mol] = hs;
  }
  std::ifstream f(dir + "/" + stem + "_mols.txt");
  std::string line;
  while (std::getline(f, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    size_t t = line.find('\t');
    p.smiles.push_back(t == std::string::npos ? "" : line.substr(t + 1));
  }
  p.fps = loadFps(dir + "/" + stem + "_ecfp.bin");
  std::printf("pool: %zu candidates\n", p.smiles.size());
  return p;
}

// Whole-PubChem table (pc_build.cpp) + popularity (pop_build.cpp, optional).
struct PubChem {
  std::vector<float> mass;
  std::vector<uint64_t> off;
  std::vector<char> smi;
  std::vector<float> pop;
  bool ok = false;
};
PubChem loadPubChem(const std::string& dir) {
  PubChem pc;
  std::vector<char> m = slurp(dir + "/pc_mass.bin", false);
  if (m.empty()) { std::printf("pubchem table: absent\n"); return pc; }
  pc.mass.resize(m.size() / 4);
  std::memcpy(pc.mass.data(), m.data(), m.size());
  std::vector<char> o = slurp(dir + "/pc_off.bin");
  pc.off.resize(o.size() / 8);
  std::memcpy(pc.off.data(), o.data(), o.size());
  pc.smi = slurp(dir + "/pc_smi.bin");
  std::vector<char> pp = slurp(dir + "/pc_pop.bin", false);
  if (!pp.empty()) { pc.pop.resize(pp.size() / 4); std::memcpy(pc.pop.data(), pp.data(), pp.size()); }
  pc.ok = true;
  std::printf("pubchem: %zu compounds, popularity %s\n", pc.mass.size(), pc.pop.empty() ? "absent" : "loaded");
  return pc;
}

RDKit::RWMol* safeParse(const std::string& smi) {
  try { return RDKit::SmilesToMol(smi); } catch (...) { return nullptr; }
}

struct Cand { float score; float rev; float pop; uint64_t fp; std::string smiles; };

}  // namespace

int main(int argc, char** argv) {
  if (argc < 7) { std::fprintf(stderr, "usage: cand_rank rank|eval ...\n"); return 1; }
  std::string mode = argv[1], dir = argv[2], qs = argv[3], modelList = argv[4];
  float ppm = float(std::atof(argv[5]));
  int topk = 25;
  std::string outPath;
  float popW;
  int skipTrain = 0;
  std::string poolStem = "pool";
  bool usePubChem = true;
  if (mode == "rank") {
    topk = std::atoi(argv[6]); outPath = argv[7]; popW = float(std::atof(argv[8]));
    if (argc > 9) skipTrain = std::atoi(argv[9]);
    if (argc > 10) poolStem = argv[10];
    if (argc > 11) usePubChem = std::atoi(argv[11]) != 0;
  } else {
    popW = float(std::atof(argv[6]));
    if (argc > 7) skipTrain = std::atoi(argv[7]);
  }

  Store q = loadSpectra(dir, qs);
  Pool pool = loadPool(dir, poolStem);
  PubChem pc = usePubChem ? loadPubChem(dir) : PubChem();
  std::vector<std::unique_ptr<Model>> models;
  for (size_t a = 0;;) {
    size_t c = modelList.find(',', a);
    std::string one = modelList.substr(a, c == std::string::npos ? std::string::npos : c - a);
    models.emplace_back(new Model());
    if (!models.back()->load(dir + "/" + one)) { std::fprintf(stderr, "cannot load %s\n", one.c_str()); return 1; }
    if (c == std::string::npos) break;
    a = c + 1;
  }
  const int NB = models[0]->NB;

  // Reverse-model bit probabilities per query molecule (mean over spectra
  // and over models).
  std::unordered_map<int, std::vector<int>> specOf;
  for (size_t s = 0; s < q.head.size(); ++s) specOf[q.head[s].mol].push_back(int(s));
  std::unordered_map<int, std::vector<float>> probs;
  for (auto& mp : models) {
    const int B = 256;
    std::vector<std::vector<std::pair<int, float>>> rows;
    std::vector<float> P;
    for (size_t s0 = 0; s0 < q.head.size(); s0 += B) {
      size_t s1 = std::min(q.head.size(), s0 + B);
      rows.assign(s1 - s0, {});
      for (size_t s = s0; s < s1; ++s) {
        const Head& h = q.head[s];
        featurize(h, q.mz.data() + h.off, q.inten.data() + h.off, rows[s - s0]);
      }
      mp->setBatch(rows, false);
      mp->forward(int(s1 - s0), false, 0.f, 0);
      download(mp->P, P, int64_t(s1 - s0) * NB);
      for (size_t s = s0; s < s1; ++s) {
        auto& v = probs[q.head[s].mol];
        if (v.empty()) v.assign(NB, 0.f);
        float w = 1.f / float(specOf[q.head[s].mol].size() * models.size());
        for (int i = 0; i < NB; ++i) v[i] += w * P[(s - s0) * NB + i];
      }
    }
  }

  std::vector<int> mols;
  for (auto& kv : specOf) mols.push_back(kv.first);
  std::sort(mols.begin(), mols.end());

  // Truth fingerprints for eval (stereo-blind match, like InChIKey14).
  std::vector<uint64_t> truthFp;
  if (mode == "eval") {
    Fps tf = loadFps(dir + "/" + qs + "_ecfp.bin");
    truthFp.resize(q.name.size());
    for (size_t i = 0; i < q.name.size(); ++i) truthFp[i] = hashBits(tf.bits.data() + tf.ptr[i], tf.ptr[i + 1] - tf.ptr[i]);
  }

  RDKit::PeriodicTable::getTable();
  std::vector<std::vector<Cand>> result(mols.size());
  std::vector<long long> nPc(mols.size(), 0), nPool(mols.size(), 0);
  std::atomic<size_t> next(0);
  unsigned nt = std::max(1u, std::thread::hardware_concurrency());
  std::vector<std::thread> th;
  for (unsigned t = 0; t < nt; ++t)
    th.emplace_back([&]() {
      std::vector<float> logit(NB);
      for (;;) {
        size_t mi = next++;
        if (mi >= mols.size()) break;
        int mol = mols[mi];
        const auto& p = probs[mol];
        for (int i = 0; i < NB; ++i) {
          float pi = std::min(1.f - 1e-6f, std::max(1e-6f, p[i]));
          logit[i] = std::log(pi / (1.f - pi));
        }
        std::vector<std::pair<float, float>> wins;  // neutral mass windows
        for (int s : specOf[mol]) {
          const Head& h = q.head[s];
          int a = h.tag < 0 ? -1 : (h.tag % 16 == 15 ? h.tag / 16 + 1 : h.tag / 16);
          float shift = (a >= 0 && a < 10) ? ADDUCT_SHIFT[a] : 1.007276f;
          float neutral = h.precursor - shift, w = neutral * ppm * 1e-6f;
          wins.emplace_back(neutral - w, neutral + w);
        }
        std::unordered_map<uint64_t, Cand> best;
        auto offer = [&](const uint16_t* b, size_t n, float pop, const std::string& smi) {
          if (n == 0) return;
          float r = 0.f;
          for (size_t k = 0; k < n; ++k) r += logit[b[k]];
          uint64_t h = hashBits(b, n);
          float sc = r + popW * pop;
          auto it = best.find(h);
          if (it == best.end()) best.emplace(h, Cand{sc, r, pop, h, smi});
          else if (sc > it->second.score) it->second = Cand{sc, r, pop, h, smi};
        };
        std::unordered_set<int> seenPool;
        for (auto& w : wins) {
          size_t lo = std::lower_bound(pool.sortedMass.begin(), pool.sortedMass.end(), w.first) - pool.sortedMass.begin();
          size_t hi = std::upper_bound(pool.sortedMass.begin(), pool.sortedMass.end(), w.second) - pool.sortedMass.begin();
          for (size_t k = lo; k < hi; ++k) {
            int c = pool.order[k];
            if (!seenPool.insert(c).second) continue;
            if (skipTrain && pool.hasSpec[c]) continue;
            ++nPool[mi];
            offer(pool.fps.bits.data() + pool.fps.ptr[c], pool.fps.ptr[c + 1] - pool.fps.ptr[c], 0.f, pool.smiles[c]);
          }
        }
        if (pc.ok) {
          std::unordered_set<size_t> seenPc;
          for (auto& w : wins) {
            size_t lo = std::lower_bound(pc.mass.begin(), pc.mass.end(), w.first) - pc.mass.begin();
            size_t hi = std::upper_bound(pc.mass.begin(), pc.mass.end(), w.second) - pc.mass.begin();
            for (size_t k = lo; k < hi; ++k) {
              if (!seenPc.insert(k).second) continue;
              std::string smi(pc.smi.data() + pc.off[k], pc.off[k + 1] - pc.off[k]);
              std::unique_ptr<RDKit::RWMol> m(safeParse(smi));
              if (!m) continue;
              auto bits = ecfp::compute(*m, 2, NB);
              ++nPc[mi];
              offer(bits.data(), bits.size(), pc.pop.empty() ? 0.f : pc.pop[k], smi);
            }
          }
        }
        auto& out = result[mi];
        out.reserve(best.size());
        for (auto& kv : best) out.push_back(kv.second);
        std::sort(out.begin(), out.end(), [](const Cand& x, const Cand& y) {
          return x.score != y.score ? x.score > y.score : x.fp < y.fp;
        });
      }
    });
  for (auto& x : th) x.join();

  long long totPc = 0, totPool = 0;
  for (size_t i = 0; i < mols.size(); ++i) { totPc += nPc[i]; totPool += nPool[i]; }
  std::printf("candidates per molecule: pool %.0f, pubchem %.0f\n",
              double(totPool) / mols.size(), double(totPc) / mols.size());

  if (mode == "eval") {
    int n = 0, top1 = 0, top5 = 0, top25 = 0, present = 0;
    double mrr = 0;
    for (size_t mi = 0; mi < mols.size(); ++mi) {
      ++n;
      uint64_t t = truthFp[mols[mi]];
      int rank = -1;
      for (size_t r = 0; r < result[mi].size(); ++r)
        if (result[mi][r].fp == t) { rank = int(r) + 1; break; }
      if (rank < 0) continue;
      ++present;
      if (rank <= 25) { mrr += 1.0 / rank; ++top25; }
      top1 += rank == 1; top5 += rank <= 5;
    }
    std::printf("eval %d mols (popW %.3f, skipTrain %d, ppm %.1f): truth among candidates %d | top1 %.1f%% top5 %.1f%% top25 %.1f%% MRR@25 %.3f\n",
                n, popW, skipTrain, ppm, present, 100.0 * top1 / n, 100.0 * top5 / n, 100.0 * top25 / n, mrr / n);
    return 0;
  }

  std::ofstream o(dir + "/" + outPath);
  // Per-candidate rev / pop scores travel with the list so the forward-model
  // cascade stage (fwd_rescore) can fuse them.
  o << "molecule_id,smiles,rev,pop\n";
  for (size_t mi = 0; mi < mols.size(); ++mi) {
    int kEnd = std::min(topk, int(result[mi].size()));
    o << q.name[mols[mi]] << ',';
    for (int k = 0; k < kEnd; ++k) o << (k ? ";" : "") << result[mi][k].smiles;
    o << ',';
    for (int k = 0; k < kEnd; ++k) o << (k ? ";" : "") << result[mi][k].rev;
    o << ',';
    for (int k = 0; k < kEnd; ++k) o << (k ? ";" : "") << result[mi][k].pop;
    o << '\n';
  }
  std::printf("wrote %s\n", outPath.c_str());
  return 0;
}
