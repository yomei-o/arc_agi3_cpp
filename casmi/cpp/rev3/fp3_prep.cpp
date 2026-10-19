// fp3_prep: vocabulary selection and bit files for fp3 (fp3.h) features.
//
//   fp3_prep count  <dir> <libstem> <counts.bin> <minCount> <threads> [kinds=127]
//       document frequency of every fp3 feature over <libstem>_mols.txt; keeps
//       count >= minCount; writes uint64 hash, uint32 count, uint8 kind pairs
//   fp3_prep select <dir> <counts.bin> <winstem> <vocab.bin> <NB> <maxWin> <perWin> <threads> [force.bin|-] [kinds]
//       information INSIDE mass windows: for windows of <winstem> (win_build
//       output; use a holdout that is NOT evaluated, e.g. haqry), the fraction
//       f of the window's candidates that carry the feature, I = mean over
//       windows of the binary entropy H(f). Keeps the NB best by I (features of
//       force.bin, an fp2/fp3 vocabulary, first). Writes an fp2-layout vocabulary
//       with rE = 300 (fp3), rF = FP3_VERSION.
//   fp3_prep apply  <dir> <mols.txt> <vocab.bin> <out_bits.bin> <threads>
//       bits of each line of <mols.txt> (name \t SMILES) for an fp2 (rE<300)
//       or fp3 vocabulary; fp_prep layout (uint32 n, nBits; uint16 count + bits)
#include <algorithm>
#include <atomic>
#include <chrono>
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
#include <GraphMol/SmilesParse/SmilesParse.h>

#include "fp3.h"

static RDKit::RWMol* safeParse(const std::string& smi) {
  try { return RDKit::SmilesToMol(smi); } catch (...) { return nullptr; }
}
static std::vector<std::string> readSmiles(const std::string& path) {
  std::vector<std::string> s;
  std::ifstream f(path);
  if (!f) { std::fprintf(stderr, "cannot read %s\n", path.c_str()); std::exit(1); }
  std::string line;
  while (std::getline(f, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    size_t t = line.find('\t');
    s.push_back(t == std::string::npos ? "" : line.substr(t + 1));
  }
  return s;
}
template <class F> static void parallelN(size_t n, int nt, F f) {
  std::atomic<size_t> next(0);
  std::vector<std::thread> th;
  for (int t = 0; t < nt; ++t)
    th.emplace_back([&]() { for (;;) { size_t i = next++; if (i >= n) break; f(i); } });
  for (auto& x : th) x.join();
}
static std::vector<uint64_t> feats(const std::string& smi, int kinds) {
  if (smi.empty()) return {};
  std::unique_ptr<RDKit::RWMol> m(safeParse(smi));
  if (!m) return {};
  try { return fp3::features(*m, kinds); } catch (...) { return {}; }
}
struct VocabFile { int32_t rE = 3, rF = 3; std::vector<uint64_t> h; };
static bool readVocab(const std::string& p, VocabFile& v) {
  std::ifstream f(p, std::ios::binary);
  if (!f) return false;
  uint32_t magic, n;
  f.read(reinterpret_cast<char*>(&magic), 4);
  f.read(reinterpret_cast<char*>(&v.rE), 4);
  f.read(reinterpret_cast<char*>(&v.rF), 4);
  f.read(reinterpret_cast<char*>(&n), 4);
  if (magic != 0x32565046) return false;
  v.h.resize(n);
  f.read(reinterpret_cast<char*>(v.h.data()), 8 * size_t(n));
  return bool(f);
}
static void writeVocab(const std::string& p, const VocabFile& v) {
  std::ofstream o(p, std::ios::binary);
  uint32_t magic = 0x32565046, n = uint32_t(v.h.size());
  o.write(reinterpret_cast<const char*>(&magic), 4);
  o.write(reinterpret_cast<const char*>(&v.rE), 4);
  o.write(reinterpret_cast<const char*>(&v.rF), 4);
  o.write(reinterpret_cast<const char*>(&n), 4);
  o.write(reinterpret_cast<const char*>(v.h.data()), 8 * size_t(n));
}

int main(int argc, char** argv) {
  if (argc < 3) { std::fprintf(stderr, "usage: fp3_prep count|select|apply ...\n"); return 1; }
  std::string mode = argv[1], dir = argv[2];
  auto t0 = std::chrono::steady_clock::now();
  auto secs = [&]() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
  if (mode == "count") {
    if (argc < 7) return 1;
    std::string stem = argv[3], outp = argv[4];
    int minCount = std::atoi(argv[5]), nt = std::atoi(argv[6]);
    int kinds = argc > 7 ? std::atoi(argv[7]) : 127;
    auto smi = readSmiles(dir + "/" + stem + "_mols.txt");
    const int NS = 64;  // sharded counting
    std::vector<std::unordered_map<uint64_t, uint32_t>> sh(NS);
    std::vector<std::mutex> mu(NS);
    std::atomic<size_t> done(0);
    parallelN(smi.size(), nt, [&](size_t i) {
      auto f = feats(smi[i], kinds);
      std::vector<std::vector<uint64_t>> part(NS);
      for (uint64_t h : f) part[h % NS].push_back(h);
      for (int s = 0; s < NS; ++s) {
        if (part[s].empty()) continue;
        std::lock_guard<std::mutex> lk(mu[s]);
        for (uint64_t h : part[s]) ++sh[s][h];
      }
      size_t d = ++done;
      if (d % 50000 == 0) { std::printf("  %zu/%zu molecules %.0fs\n", d, smi.size(), secs()); std::fflush(stdout); }
    });
    size_t tot = 0, kept = 0;
    std::ofstream o(dir + "/" + outp, std::ios::binary);
    for (auto& m : sh)
      for (auto& kv : m) {
        ++tot;
        if (int(kv.second) < minCount) continue;
        ++kept;
        o.write(reinterpret_cast<const char*>(&kv.first), 8);
        o.write(reinterpret_cast<const char*>(&kv.second), 4);
      }
    std::printf("count %s: %zu molecules, %zu distinct features, kept %zu with count >= %d (%.0fs)\n", stem.c_str(),
                smi.size(), tot, kept, minCount, secs());
    return 0;
  }
  if (mode == "select") {
    if (argc < 10) return 1;
    std::string cntp = argv[3], wst = argv[4], vout = argv[5];
    int NB = std::atoi(argv[6]), maxWin = std::atoi(argv[7]), perWin = std::atoi(argv[8]), nt = std::atoi(argv[9]);
    std::string force = argc > 10 ? argv[10] : "-";
    int kinds = argc > 11 ? std::atoi(argv[11]) : 127;
    std::unordered_map<uint64_t, uint32_t> cnt;
    {
      std::ifstream f(dir + "/" + cntp, std::ios::binary);
      uint64_t h; uint32_t c;
      while (f.read(reinterpret_cast<char*>(&h), 8) && f.read(reinterpret_cast<char*>(&c), 4)) cnt[h] = c;
    }
    std::printf("select: %zu counted features\n", cnt.size());
    // windows
    std::vector<std::vector<uint32_t>> wins;
    {
      std::ifstream f(dir + "/" + wst + "_win.bin", std::ios::binary);
      uint32_t nq; f.read(reinterpret_cast<char*>(&nq), 4);
      for (uint32_t q = 0; q < nq; ++q) {
        int32_t m, t; uint32_t n;
        f.read(reinterpret_cast<char*>(&m), 4); f.read(reinterpret_cast<char*>(&t), 4); f.read(reinterpret_cast<char*>(&n), 4);
        std::vector<uint32_t> ids(n);
        if (n) f.read(reinterpret_cast<char*>(ids.data()), 4 * size_t(n));
        if (n < 20) continue;
        if (int(ids.size()) > perWin) {  // deterministic stride sample
          std::vector<uint32_t> s;
          for (int k = 0; k < perWin; ++k) s.push_back(ids[size_t(k) * ids.size() / perWin]);
          ids.swap(s);
        }
        wins.push_back(std::move(ids));
        if (int(wins.size()) >= maxWin) break;
      }
    }
    auto smi = readSmiles(dir + "/" + wst + "_wmols.txt");
    std::unordered_map<uint32_t, int> need;
    std::vector<uint32_t> uids;
    for (auto& w : wins) for (uint32_t u : w) if (need.emplace(u, int(uids.size())).second) uids.push_back(u);
    std::printf("select: %zu windows, %zu unique molecules to featurize\n", wins.size(), uids.size());
    std::fflush(stdout);
    // feature index over counted features
    std::vector<uint64_t> fh;
    std::unordered_map<uint64_t, int> fi;
    for (auto& kv : cnt) { fi.emplace(kv.first, int(fh.size())); fh.push_back(kv.first); }
    std::vector<std::vector<int>> mf(uids.size());
    parallelN(uids.size(), nt, [&](size_t i) {
      for (uint64_t h : feats(smi[uids[i]], kinds)) {
        auto it = fi.find(h);
        if (it != fi.end()) mf[i].push_back(it->second);
      }
    });
    std::printf("select: featurized (%.0fs)\n", secs());
    std::vector<double> I(fh.size(), 0.0);
    std::vector<int> c(fh.size(), 0);
    for (auto& w : wins) {
      std::vector<int> touched;
      for (uint32_t u : w)
        for (int f : mf[need[u]]) { if (c[f]++ == 0) touched.push_back(f); }
      double n = double(w.size());
      for (int f : touched) {
        double p = c[f] / n;
        if (p > 0 && p < 1) I[f] += -(p * std::log2(p) + (1 - p) * std::log2(1 - p));
        c[f] = 0;
      }
    }
    for (auto& x : I) x /= double(std::max<size_t>(1, wins.size()));
    VocabFile fv;
    std::unordered_set<uint64_t> chosen;
    VocabFile out;
    out.rE = 300; out.rF = fp3::FP3_VERSION;
    if (force != "-") {
      if (!readVocab(dir + "/" + force, fv)) { std::fprintf(stderr, "bad force vocab\n"); return 1; }
      for (uint64_t h : fv.h) if (chosen.insert(h).second) out.h.push_back(h);
      std::printf("select: forced %zu features of %s\n", out.h.size(), force.c_str());
    }
    std::vector<int> ord(fh.size());
    for (size_t i = 0; i < ord.size(); ++i) ord[i] = int(i);
    std::sort(ord.begin(), ord.end(), [&](int a, int b) { return I[a] != I[b] ? I[a] > I[b] : fh[a] < fh[b]; });
    double sumI = 0;
    for (int f : ord) {
      if (int(out.h.size()) >= NB) break;
      if (I[f] <= 0) break;
      if (chosen.insert(fh[f]).second) { out.h.push_back(fh[f]); sumI += I[f]; }
    }
    // information of the forced part, for the log
    double forcedI = 0;
    for (uint64_t h : fv.h) { auto it = fi.find(h); if (it != fi.end()) forcedI += I[it->second]; }
    std::printf("select: vocab %zu (NB %d); in-window entropy sum: forced %.1f, added %.1f; I of rank 1/100/1000/last added: %.3f %.3f %.3f\n",
                out.h.size(), NB, forcedI, sumI, I[ord[0]], I[ord[std::min<size_t>(99, ord.size() - 1)]],
                I[ord[std::min<size_t>(999, ord.size() - 1)]], out.h.empty() ? 0.0 : 0.0);
    writeVocab(dir + "/" + vout, out);
    return 0;
  }
  if (mode == "apply") {
    if (argc < 7) return 1;
    std::string molsp = argv[3], vp = argv[4], outp = argv[5];
    int nt = std::atoi(argv[6]);
    VocabFile v;
    if (!readVocab(dir + "/" + vp, v)) { std::fprintf(stderr, "bad vocab %s\n", vp.c_str()); return 1; }
    std::unordered_map<uint64_t, int> idx;
    for (size_t i = 0; i < v.h.size(); ++i) idx.emplace(v.h[i], int(i));
    auto smi = readSmiles(dir + "/" + molsp);
    std::vector<std::vector<uint16_t>> out(smi.size());
    const bool isFp3 = v.rE >= 300;
    std::atomic<size_t> done(0);
    parallelN(smi.size(), nt, [&](size_t i) {
      if (smi[i].empty()) return;
      std::unique_ptr<RDKit::RWMol> m(safeParse(smi[i]));
      if (!m) return;
      std::vector<uint64_t> f;
      try { f = isFp3 ? fp3::features(*m) : fp2::envs(*m, v.rE, v.rF); } catch (...) { return; }
      for (uint64_t h : f) { auto it = idx.find(h); if (it != idx.end()) out[i].push_back(uint16_t(it->second)); }
      std::sort(out[i].begin(), out[i].end());
      size_t d = ++done;
      if (d % 500000 == 0) { std::printf("  %zu/%zu %.0fs\n", d, smi.size(), secs()); std::fflush(stdout); }
    });
    std::ofstream o(dir + "/" + outp, std::ios::binary);
    uint32_t n = uint32_t(out.size()), nb = uint32_t(v.h.size());
    o.write(reinterpret_cast<const char*>(&n), 4);
    o.write(reinterpret_cast<const char*>(&nb), 4);
    double mean = 0;
    size_t empty = 0;
    for (auto& b : out) {
      uint16_t c = uint16_t(std::min<size_t>(b.size(), 65535));
      o.write(reinterpret_cast<const char*>(&c), 2);
      if (c) o.write(reinterpret_cast<const char*>(b.data()), 2 * size_t(c));
      mean += c; empty += c == 0;
    }
    std::printf("apply %s -> %s: %u molecules, %u bits, mean %.1f on, %zu empty (%.0fs)\n", molsp.c_str(), outp.c_str(), n,
                nb, mean / std::max<size_t>(1, out.size()), empty, secs());
    return 0;
  }
  std::fprintf(stderr, "unknown mode\n");
  return 1;
}
