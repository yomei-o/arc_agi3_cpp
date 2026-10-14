// Merge pubchem_parse.cpp's new-structure output into the existing
// pool_head.bin/pool_mols.txt pair, same binary layout build_pool.py used
// for the COCONUT merge (mono_mass f32, mol_id i32, has_spectrum i32 per
// record in pool_head.bin, mass-sorted; name\tsmiles per line in
// pool_mols.txt, indexed by mol_id). New entries get has_spectrum=0, same as
// COCONUT's did - they're structure-only candidates, no reference spectrum.
// No RDKit needed here: pubchem_parse.cpp already computed mass and
// deduplicated against this exact pool_mols.txt by canonical SMILES, so this
// step is pure read-append-sort-write.
//
// Usage:
//   pool_merge <root with pool_head.bin/pool_mols.txt> <pubchem_new.tsv>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

struct Rec { float mass; int32_t mol; int32_t hasSpec; };

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <root> <pubchem_new.tsv>\n", argv[0]);
    return 1;
  }
  std::string root = argv[1];

  std::vector<std::string> names;
  std::vector<std::string> smiles;
  {
    std::ifstream f(root + "/pool_mols.txt");
    if (!f) { std::fprintf(stderr, "cannot read pool_mols.txt\n"); return 1; }
    std::string line;
    while (std::getline(f, line)) {
      if (!line.empty() && line.back() == '\r') line.pop_back();
      size_t t = line.find('\t');
      names.push_back(line.substr(0, t));
      smiles.push_back(t == std::string::npos ? "" : line.substr(t + 1));
    }
  }
  size_t nOld = names.size();
  std::fprintf(stderr, "existing pool_mols.txt: %zu entries\n", nOld);

  std::vector<Rec> recs;
  {
    std::ifstream f(root + "/pool_head.bin", std::ios::binary | std::ios::ate);
    if (!f) { std::fprintf(stderr, "cannot read pool_head.bin\n"); return 1; }
    size_t n = size_t(f.tellg()) / sizeof(Rec);
    f.seekg(0);
    recs.resize(n);
    f.read(reinterpret_cast<char*>(recs.data()), n * sizeof(Rec));
  }
  std::fprintf(stderr, "existing pool_head.bin: %zu records\n", recs.size());

  long long nNew = 0;
  {
    std::ifstream f(argv[2]);
    if (!f) { std::fprintf(stderr, "cannot read %s\n", argv[2]); return 1; }
    std::string line;
    while (std::getline(f, line)) {
      if (!line.empty() && line.back() == '\r') line.pop_back();
      size_t t1 = line.find('\t');
      size_t t2 = line.find('\t', t1 + 1);
      if (t1 == std::string::npos || t2 == std::string::npos) continue;
      std::string name = line.substr(0, t1);
      std::string smi = line.substr(t1 + 1, t2 - t1 - 1);
      float mass = float(std::atof(line.c_str() + t2 + 1));
      int32_t molId = int32_t(names.size());
      names.push_back(name);
      smiles.push_back(smi);
      recs.push_back({mass, molId, 0});
      ++nNew;
    }
  }
  std::fprintf(stderr, "added %lld new structures, pool total %zu\n", nNew, names.size());

  std::sort(recs.begin(), recs.end(), [](const Rec& a, const Rec& b) { return a.mass < b.mass; });

  std::ofstream hf(root + "/pool_head.bin", std::ios::binary | std::ios::trunc);
  hf.write(reinterpret_cast<const char*>(recs.data()), recs.size() * sizeof(Rec));
  hf.close();

  std::ofstream mf(root + "/pool_mols.txt", std::ios::trunc);
  for (size_t i = 0; i < names.size(); ++i) {
    if (i) mf << '\n';
    mf << names[i] << '\t' << smiles[i];
  }
  mf.close();

  std::fprintf(stderr, "wrote pool_head.bin (%zu records, mass-sorted) and pool_mols.txt (%zu entries, %zu were already there)\n",
               recs.size(), names.size(), nOld);
  return 0;
}
