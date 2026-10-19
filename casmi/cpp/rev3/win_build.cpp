// win_build: cache of the PubChem mass windows of a query set (the "window
// MRR" yardstick of fp3_STATUS.md).
//
//   win_build <dir> <qry> <pcdir> <out_stem> [ppm=10] [threads=8] [sampleK=0] [seed=1]
//   (sampleK > 0: only ~sampleK random compounds of each window = training decoys)
//
// For each query molecule of <qry> (<qry>_head.bin / _mols.txt = name \t truth
// SMILES): one neutral-mass window per spectrum (same adduct shifts as
// phys_pc / phys_v28pc: 2M adducts halve the mass), +-ppm, merged, read from
// the mass-sorted PubChem table (pc_mass.bin / pc_off.bin / pc_smi.bin /
// pc_pop.bin of pc_build.cpp / pop_build.cpp). Every compound is parsed and
// keyed by its stereo-blind canonical SMILES (MolToSmiles(non-isomeric),
// re-parsed once, as phys_v28pc's canon()); compounds with the same key are
// one candidate (pop = max over their CIDs), so stereoisomers do not push
// the truth down. truth = candidate whose key equals the truth's key.
//
// Output:
//   <out>_wmols.txt   uid \t key SMILES     (unique candidates over all queries)
//   <out>_wpop.bin    float32 per uid
//   <out>_win.bin     uint32 nQ; per query: int32 mol (row of <qry>_mols.txt),
//                     int32 truth position in the list (-1 = not in window),
//                     uint32 n, n x uint32 uid
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <GraphMol/GraphMol.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/SmilesParse/SmilesWrite.h>

struct Head { float precursor; int32_t mol, off, n, tag; };
constexpr float ADDUCT_SHIFT[10] = {
    1.007276f, -1.007276f, 22.989218f, 18.033823f, 38.963158f,
    34.969402f, 44.998201f, -17.002740f, 1.007276f, 22.989218f,
};
static int adductIdx(int32_t tag) {
  if (tag < 0) return -1;
  return tag % 16 == 15 ? tag / 16 + 1 : tag / 16;
}

static std::vector<char> slurp(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) { std::fprintf(stderr, "cannot read %s\n", path.c_str()); std::exit(1); }
  std::streamsize n = f.tellg();
  f.seekg(0);
  std::vector<char> b((size_t(n)));
  f.read(b.data(), n);
  return b;
}
template <class T> static void rd(const std::string& p, std::vector<T>& v) {
  auto b = slurp(p);
  v.resize(b.size() / sizeof(T));
  std::memcpy(v.data(), b.data(), v.size() * sizeof(T));
}

static std::string canonKey(const std::string& smi) {
  std::unique_ptr<RDKit::RWMol> m;
  try { m.reset(RDKit::SmilesToMol(smi)); } catch (...) { m.reset(); }
  if (!m) return std::string();
  std::string k;
  try { k = RDKit::MolToSmiles(*m, false); } catch (...) { return std::string(); }
  std::unique_ptr<RDKit::RWMol> r;
  try { r.reset(RDKit::SmilesToMol(k)); } catch (...) { r.reset(); }
  if (!r) return k;
  try { return RDKit::MolToSmiles(*r, false); } catch (...) { return k; }
}

int main(int argc, char** argv) {
  if (argc < 5) { std::fprintf(stderr, "usage: win_build <dir> <qry> <pcdir> <out_stem> [ppm] [threads]\n"); return 1; }
  std::string dir = argv[1], qry = argv[2], pcdir = argv[3], out = argv[4];
  double ppm = argc > 5 ? std::atof(argv[5]) : 10.0;
  int nThr = argc > 6 ? std::atoi(argv[6]) : 8;
  // sampleK > 0: keep only ~sampleK random compounds per window (decoys for training)
  const uint64_t sampleK = argc > 7 ? uint64_t(std::atoll(argv[7])) : 0;
  const uint64_t seed = argc > 8 ? uint64_t(std::atoll(argv[8])) : 1;

  std::vector<Head> head;
  rd(dir + "/" + qry + "_head.bin", head);
  std::vector<std::string> tsmi;
  {
    std::ifstream f(dir + "/" + qry + "_mols.txt");
    std::string line;
    while (std::getline(f, line)) {
      if (!line.empty() && line.back() == '\r') line.pop_back();
      size_t t = line.find('\t');
      tsmi.push_back(t == std::string::npos ? "" : line.substr(t + 1));
    }
  }
  const int nQ = int(tsmi.size());
  std::vector<std::vector<int>> specOf(nQ);
  for (size_t s = 0; s < head.size(); ++s)
    if (head[s].mol >= 0 && head[s].mol < nQ && head[s].n > 0) specOf[head[s].mol].push_back(int(s));

  std::vector<float> pcMass, pcPop;
  std::vector<uint64_t> pcOff;
  rd(pcdir + "/pc_mass.bin", pcMass);
  rd(pcdir + "/pc_off.bin", pcOff);
  rd(pcdir + "/pc_pop.bin", pcPop);
  std::printf("%s: %d query molecules, %zu spectra; pubchem %zu compounds; ppm %g, threads %d\n", qry.c_str(), nQ,
              head.size(), pcMass.size(), ppm, nThr);
  std::fflush(stdout);

  struct QOut { std::vector<std::string> keys; std::vector<float> pop; int truth = -1; size_t raw = 0; };
  std::vector<QOut> res(nQ);
  std::atomic<int> next(0), done(0);
  auto t0 = std::chrono::steady_clock::now();
  std::vector<std::thread> th;
  for (int t = 0; t < nThr; ++t)
    th.emplace_back([&]() {
      std::ifstream fs(pcdir + "/pc_smi.bin", std::ios::binary);
      std::string buf;
      for (;;) {
        int q = next++;
        if (q >= nQ) break;
        QOut& o = res[q];
        std::vector<std::pair<size_t, size_t>> rng;
        for (int s : specOf[q]) {
          const Head& h = head[s];
          int a = adductIdx(h.tag);
          float shift = (a >= 0 && a < 10) ? ADDUCT_SHIFT[a] : 1.007276f;
          float neutral = h.precursor - shift;
          if (a == 8 || a == 9) neutral *= 0.5f;
          float w = float(neutral * ppm * 1e-6);
          size_t lo = std::lower_bound(pcMass.begin(), pcMass.end(), neutral - w) - pcMass.begin();
          size_t hi = std::upper_bound(pcMass.begin(), pcMass.end(), neutral + w) - pcMass.begin();
          if (hi > lo) rng.emplace_back(lo, hi);
        }
        std::sort(rng.begin(), rng.end());
        std::vector<std::pair<size_t, size_t>> mr;
        for (auto& r : rng) {
          if (!mr.empty() && r.first <= mr.back().second) mr.back().second = std::max(mr.back().second, r.second);
          else mr.push_back(r);
        }
        std::string tkey = tsmi[q].empty() ? std::string() : canonKey(tsmi[q]);
        std::unordered_map<std::string, int> idx;
        uint64_t R = 0;
        for (auto& r : mr) R += r.second - r.first;
        for (auto& r : mr) {
          uint64_t o0 = pcOff[r.first], o1 = pcOff[r.second];
          buf.resize(size_t(o1 - o0));
          fs.seekg(std::streamoff(o0));
          fs.read(buf.data(), std::streamsize(o1 - o0));
          for (size_t k = r.first; k < r.second; ++k) {
            ++o.raw;
            if (sampleK > 0 && R > sampleK) {
              uint64_t x = (uint64_t(k) + seed * 0x9E3779B97F4A7C15ull) * 0xBF58476D1CE4E5B9ull;
              x ^= x >> 31; x *= 0x94D049BB133111EBull; x ^= x >> 29;
              if (x % R >= sampleK) continue;
            }
            std::string key = canonKey(buf.substr(size_t(pcOff[k] - o0), size_t(pcOff[k + 1] - pcOff[k])));
            if (key.empty()) continue;
            auto it = idx.find(key);
            if (it == idx.end()) {
              idx.emplace(key, int(o.keys.size()));
              if (!tkey.empty() && key == tkey) o.truth = int(o.keys.size());
              o.keys.push_back(key);
              o.pop.push_back(pcPop[k]);
            } else {
              o.pop[it->second] = std::max(o.pop[it->second], pcPop[k]);
            }
          }
        }
        int d = ++done;
        if (d % 50 == 0) {
          double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
          std::printf("  %d/%d queries, %.0fs\n", d, nQ, sec);
          std::fflush(stdout);
        }
      }
    });
  for (auto& x : th) x.join();

  std::unordered_map<std::string, uint32_t> uid;
  std::vector<const std::string*> ukeys;
  std::vector<float> upop;
  std::ofstream ow(dir + "/" + out + "_win.bin", std::ios::binary);
  uint32_t nq = uint32_t(nQ);
  ow.write(reinterpret_cast<const char*>(&nq), 4);
  int inWin = 0;
  double sz = 0, raw = 0;
  for (int q = 0; q < nQ; ++q) {
    QOut& o = res[q];
    std::vector<uint32_t> ids(o.keys.size());
    for (size_t i = 0; i < o.keys.size(); ++i) {
      auto it = uid.find(o.keys[i]);
      if (it == uid.end()) {
        it = uid.emplace(o.keys[i], uint32_t(ukeys.size())).first;
        ukeys.push_back(&it->first);
        upop.push_back(o.pop[i]);
      } else {
        upop[it->second] = std::max(upop[it->second], o.pop[i]);
      }
      ids[i] = it->second;
    }
    int32_t m = q, t = o.truth;
    uint32_t n = uint32_t(ids.size());
    ow.write(reinterpret_cast<const char*>(&m), 4);
    ow.write(reinterpret_cast<const char*>(&t), 4);
    ow.write(reinterpret_cast<const char*>(&n), 4);
    if (n) ow.write(reinterpret_cast<const char*>(ids.data()), 4 * size_t(n));
    inWin += o.truth >= 0;
    sz += double(n);
    raw += double(o.raw);
  }
  {
    std::ofstream om(dir + "/" + out + "_wmols.txt");
    for (size_t i = 0; i < ukeys.size(); ++i) om << i << '\t' << *ukeys[i] << '\n';
    std::ofstream op(dir + "/" + out + "_wpop.bin", std::ios::binary);
    op.write(reinterpret_cast<const char*>(upop.data()), 4 * upop.size());
  }
  double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("%s: truth in window %d/%d (%.3f); window mean %.0f unique keys (%.0f raw compounds); %zu unique over all; %.0fs\n",
              qry.c_str(), inWin, nQ, double(inWin) / std::max(1, nQ), sz / std::max(1, nQ), raw / std::max(1, nQ),
              ukeys.size(), sec);
  return 0;
}
