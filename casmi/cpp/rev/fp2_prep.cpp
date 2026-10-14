// fp2_prep: vocabulary fingerprints (fp2.h) for the reverse model v2.
//
//   fp2_prep build <dir> <libstem> <vocab.bin> <rE> <rF> <maxFeat> <minCount>
//       count environment document frequencies over <libstem>_mols.txt, keep
//       the <maxFeat> most frequent with count >= minCount, write the vocab
//       and <libstem>_<vocabtag>.bin (vocabtag = vocab file name without .bin)
//   fp2_prep apply <dir> <stem> <vocab.bin>
//       write <stem>_<vocabtag>.bin
//   fp2_prep cands <dir> <pool_stem> <out_stem> <features.txt>...
//       collect every candidate key of phys_ana feature dumps, write
//       <out_stem>_mols.txt (key \t SMILES from <pool_stem>_mols.txt)
//
// Bits files use fp_prep's layout: uint32 nMols, uint32 nBits, then per
// molecule uint16 count + count x uint16 bit index.
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <GraphMol/GraphMol.h>
#include <GraphMol/SmilesParse/SmilesParse.h>

#include "fp2.h"

static RDKit::RWMol* safeParse(const std::string& smi) {
  try { return RDKit::SmilesToMol(smi); } catch (...) { return nullptr; }
}

static std::vector<std::string> readSmiles(const std::string& path, std::vector<std::string>* names = nullptr) {
  std::vector<std::string> s;
  std::ifstream f(path);
  if (!f) { std::fprintf(stderr, "cannot read %s\n", path.c_str()); std::exit(1); }
  std::string line;
  while (std::getline(f, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    size_t t = line.find('\t');
    s.push_back(t == std::string::npos ? "" : line.substr(t + 1));
    if (names) names->push_back(line.substr(0, t));
  }
  return s;
}

static std::string tagOf(const std::string& vocabPath) {
  std::string t = vocabPath;
  size_t sl = t.find_last_of("/\\");
  if (sl != std::string::npos) t = t.substr(sl + 1);
  if (t.size() > 4 && t.substr(t.size() - 4) == ".bin") t.resize(t.size() - 4);
  return t;
}

template <class F>
static void parallelMols(size_t n, F f) {
  std::atomic<size_t> next(0);
  unsigned nt = std::max(1u, std::thread::hardware_concurrency());
  std::vector<std::thread> th;
  for (unsigned t = 0; t < nt; ++t)
    th.emplace_back([&]() {
      for (;;) {
        size_t i = next++;
        if (i >= n) break;
        f(i);
      }
    });
  for (auto& x : th) x.join();
}

static void writeBits(const std::string& path, const std::vector<std::vector<uint16_t>>& out, uint32_t nBits) {
  std::ofstream o(path, std::ios::binary);
  uint32_t n = uint32_t(out.size());
  o.write(reinterpret_cast<const char*>(&n), 4);
  o.write(reinterpret_cast<const char*>(&nBits), 4);
  double mean = 0;
  for (auto& v : out) {
    uint16_t c = uint16_t(v.size());
    o.write(reinterpret_cast<const char*>(&c), 2);
    if (c) o.write(reinterpret_cast<const char*>(v.data()), 2 * c);
    mean += c;
  }
  std::fprintf(stderr, "wrote %s: %u molecules, %u bits, mean %.1f on\n", path.c_str(), n, nBits,
               mean / std::max<size_t>(1, out.size()));
}

static bool loadVocab(const std::string& path, fp2::Vocab& v) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  uint32_t magic, n;
  int32_t rE, rF;
  f.read(reinterpret_cast<char*>(&magic), 4);
  f.read(reinterpret_cast<char*>(&rE), 4);
  f.read(reinterpret_cast<char*>(&rF), 4);
  f.read(reinterpret_cast<char*>(&n), 4);
  if (magic != 0x32565046) return false;
  v.rE = rE; v.rF = rF;
  v.hashes.resize(n);
  f.read(reinterpret_cast<char*>(v.hashes.data()), 8 * size_t(n));
  v.build();
  return true;
}

int main(int argc, char** argv) {
  if (argc < 5) { std::fprintf(stderr, "usage: fp2_prep build|apply|cands ...\n"); return 1; }
  std::string mode = argv[1], dir = argv[2];
  if (mode == "cands") {
    std::string poolStem = argv[3], outStem = argv[4];
    std::set<std::string> keys;
    for (int a = 5; a < argc; ++a) {
      std::ifstream f(dir + "/" + argv[a]);
      if (!f) { std::fprintf(stderr, "cannot read %s\n", argv[a]); return 1; }
      std::string line;
      while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == 'Q') continue;
        size_t sp = line.find_last_of(' ');
        keys.insert(line.substr(sp + 1));
      }
    }
    std::vector<std::string> names;
    auto smi = readSmiles(dir + "/" + poolStem + "_mols.txt", &names);
    std::ofstream o(dir + "/" + outStem + "_mols.txt");
    size_t found = 0;
    std::set<std::string> done;
    for (size_t i = 0; i < names.size(); ++i)
      if (keys.count(names[i]) && done.insert(names[i]).second) { o << names[i] << '\t' << smi[i] << '\n'; ++found; }
    std::fprintf(stderr, "%zu candidate keys, %zu written\n", keys.size(), found);
    return 0;
  }
  if (mode == "build") {
    if (argc < 9) { std::fprintf(stderr, "usage: fp2_prep build <dir> <libstem> <vocab.bin> <rE> <rF> <maxFeat> <minCount>\n"); return 1; }
    std::string stem = argv[3], vocabPath = argv[4];
    int rE = std::atoi(argv[5]), rF = std::atoi(argv[6]);
    size_t maxFeat = size_t(std::atoll(argv[7]));
    int minCount = std::atoi(argv[8]);
    auto smi = readSmiles(dir + "/" + stem + "_mols.txt");
    std::vector<std::vector<uint64_t>> env(smi.size());
    parallelMols(smi.size(), [&](size_t i) {
      if (smi[i].empty()) return;
      std::unique_ptr<RDKit::RWMol> m(safeParse(smi[i]));
      if (m) env[i] = fp2::envs(*m, rE, rF);
    });
    std::unordered_map<uint64_t, int> cnt;
    for (auto& e : env) for (uint64_t h : e) ++cnt[h];
    std::vector<std::pair<int, uint64_t>> c;
    for (auto& kv : cnt) if (kv.second >= minCount) c.emplace_back(kv.second, kv.first);
    std::sort(c.begin(), c.end(), [](auto& a, auto& b) { return a.first != b.first ? a.first > b.first : a.second < b.second; });
    if (c.size() > maxFeat) c.resize(maxFeat);
    std::fprintf(stderr, "%zu distinct environments, kept %zu (count %d .. %d of %zu mols)\n", cnt.size(), c.size(),
                 c.empty() ? 0 : c.back().first, c.empty() ? 0 : c.front().first, smi.size());
    fp2::Vocab v;
    v.rE = rE; v.rF = rF;
    for (auto& x : c) v.hashes.push_back(x.second);
    v.build();
    {
      std::ofstream o(dir + "/" + vocabPath, std::ios::binary);
      uint32_t magic = 0x32565046, n = uint32_t(v.hashes.size());
      int32_t a = rE, b = rF;
      o.write(reinterpret_cast<const char*>(&magic), 4);
      o.write(reinterpret_cast<const char*>(&a), 4);
      o.write(reinterpret_cast<const char*>(&b), 4);
      o.write(reinterpret_cast<const char*>(&n), 4);
      o.write(reinterpret_cast<const char*>(v.hashes.data()), 8 * size_t(n));
    }
    std::vector<std::vector<uint16_t>> out(smi.size());
    for (size_t i = 0; i < smi.size(); ++i) {
      for (uint64_t h : env[i]) {
        auto it = v.index.find(h);
        if (it != v.index.end()) out[i].push_back(uint16_t(it->second));
      }
      std::sort(out[i].begin(), out[i].end());
    }
    writeBits(dir + "/" + stem + "_" + tagOf(vocabPath) + ".bin", out, uint32_t(v.hashes.size()));
    return 0;
  }
  if (mode == "apply") {
    std::string stem = argv[3], vocabPath = argv[4];
    fp2::Vocab v;
    if (!loadVocab(dir + "/" + vocabPath, v)) { std::fprintf(stderr, "bad vocab %s\n", vocabPath.c_str()); return 1; }
    auto smi = readSmiles(dir + "/" + stem + "_mols.txt");
    std::vector<std::vector<uint16_t>> out(smi.size());
    parallelMols(smi.size(), [&](size_t i) {
      if (smi[i].empty()) return;
      std::unique_ptr<RDKit::RWMol> m(safeParse(smi[i]));
      if (m) out[i] = v.bits(*m);
    });
    writeBits(dir + "/" + stem + "_" + tagOf(vocabPath) + ".bin", out, uint32_t(v.hashes.size()));
    return 0;
  }
  std::fprintf(stderr, "unknown mode\n");
  return 1;
}
