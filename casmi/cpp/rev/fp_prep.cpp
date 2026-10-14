// Compute ecfp::compute for every molecule in <stem>_mols.txt (name\tsmiles,
// the layout export_bin.py / build_pool.py / pool_merge.cpp all write) and
// store the on-bits, one variable-length record per line in mols.txt order
// so a mol_id indexes both files the same way.
//
// <stem>_ecfp.bin: uint32 nMols, uint32 nBits, then per molecule
//   uint16 count, count x uint16 bit index (count 0 = SMILES failed to parse
//   or was empty - a molecule nothing can be scored against).
//
// Usage: fp_prep <dir> <stem> [radius] [nBits]
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <GraphMol/GraphMol.h>
#include <GraphMol/SmilesParse/SmilesParse.h>

#include "ecfp.h"

static RDKit::RWMol* safeParse(const std::string& smi) {
  try { return RDKit::SmilesToMol(smi); } catch (...) { return nullptr; }
}

int main(int argc, char** argv) {
  if (argc < 3) { std::fprintf(stderr, "usage: %s <dir> <stem> [radius] [nBits]\n", argv[0]); return 1; }
  std::string dir = argv[1], stem = argv[2];
  int radius = argc > 3 ? std::atoi(argv[3]) : 2;
  int nBits = argc > 4 ? std::atoi(argv[4]) : 2048;

  std::vector<std::string> smiles;
  {
    std::ifstream f(dir + "/" + stem + "_mols.txt");
    if (!f) { std::fprintf(stderr, "cannot read %s_mols.txt\n", stem.c_str()); return 1; }
    std::string line;
    while (std::getline(f, line)) {
      if (!line.empty() && line.back() == '\r') line.pop_back();
      size_t t = line.find('\t');
      smiles.push_back(t == std::string::npos ? "" : line.substr(t + 1));
    }
  }
  std::fprintf(stderr, "%s: %zu molecules\n", stem.c_str(), smiles.size());

  std::vector<std::vector<uint16_t>> out(smiles.size());
  std::atomic<size_t> next(0), nFail(0);
  unsigned nt = std::max(1u, std::thread::hardware_concurrency());
  std::vector<std::thread> th;
  for (unsigned t = 0; t < nt; ++t)
    th.emplace_back([&]() {
      for (;;) {
        size_t i = next++;
        if (i >= smiles.size()) break;
        if (smiles[i].empty()) { ++nFail; continue; }
        std::unique_ptr<RDKit::RWMol> m(safeParse(smiles[i]));
        if (!m) { ++nFail; continue; }
        out[i] = ecfp::compute(*m, radius, nBits);
        if (i % 500000 == 0) std::fprintf(stderr, "  %zu\n", i);
      }
    });
  for (auto& x : th) x.join();

  std::ofstream o(dir + "/" + stem + "_ecfp.bin", std::ios::binary);
  uint32_t n = uint32_t(out.size()), nb = uint32_t(nBits);
  o.write(reinterpret_cast<const char*>(&n), 4);
  o.write(reinterpret_cast<const char*>(&nb), 4);
  double meanBits = 0;
  for (auto& v : out) {
    uint16_t c = uint16_t(v.size());
    o.write(reinterpret_cast<const char*>(&c), 2);
    if (c) o.write(reinterpret_cast<const char*>(v.data()), 2 * c);
    meanBits += c;
  }
  std::fprintf(stderr, "wrote %s_ecfp.bin: %u molecules, %zu failed, mean %.1f on-bits\n",
               stem.c_str(), n, nFail.load(), meanBits / std::max<size_t>(1, out.size()));
  return 0;
}
