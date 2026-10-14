// Stage 2: pull the SMILES for exactly the CIDs that survived the mass
// filter (pubchem_filter.cpp's output) out of PubChem's full CID-SMILES
// file, without RDKit and without holding the whole ~1.1e8-row file's
// SMILES in memory - only the surviving CIDs need to be, as a hash set of
// integers (a few million CIDs is tens of MB, nothing like the full file).
//
// Usage:
//   pubchem_join <filtered_mass.tsv> <CID-SMILES.tsv> <out.tsv>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <unordered_map>

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: %s <filtered_mass.tsv> <CID-SMILES.tsv> <out.tsv>\n", argv[0]);
    return 1;
  }
  std::unordered_map<long long, float> wantedMass;
  {
    std::ifstream in(argv[1]);
    if (!in) { std::fprintf(stderr, "cannot read %s\n", argv[1]); return 1; }
    std::string line;
    while (std::getline(in, line)) {
      size_t tab = line.find('\t');
      if (tab == std::string::npos) continue;
      long long cid = std::atoll(line.c_str());
      float mass = float(std::atof(line.c_str() + tab + 1));
      wantedMass[cid] = mass;
    }
  }
  std::fprintf(stderr, "looking for %zu CIDs\n", wantedMass.size());

  std::ifstream in(argv[2]);
  if (!in) { std::fprintf(stderr, "cannot read %s\n", argv[2]); return 1; }
  std::ofstream out(argv[3]);
  out << "cid\tmass\tsmiles\n";

  std::string line;
  long long total = 0, found = 0;
  while (std::getline(in, line)) {
    ++total;
    size_t tab = line.find('\t');
    if (tab == std::string::npos) continue;
    long long cid = std::atoll(line.c_str());
    auto it = wantedMass.find(cid);
    if (it != wantedMass.end()) {
      out << cid << '\t' << it->second << '\t' << line.substr(tab + 1) << '\n';
      ++found;
      if (found == wantedMass.size()) break;  // got them all, stop reading early
    }
    if (total % 20000000 == 0)
      std::fprintf(stderr, "  %lld lines scanned, %lld/%zu found\n", total, found, wantedMass.size());
  }
  std::fprintf(stderr, "done: %lld lines scanned, %lld/%zu CIDs matched\n",
               total, found, wantedMass.size());
  return 0;
}
