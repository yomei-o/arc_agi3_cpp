// Diagnostic: find which pool_mols.txt line crashes RDKit's minimal
// SmilesToMol, by printing (and flushing) the index immediately before each
// parse. pubchem_parse.cpp died at the same ~400000-500000 point on two
// separate runs with no pipe/ssh interruption involved the second time, so
// this isolates whether it's one specific pathological SMILES.
//
// Usage: smiles_scan <pool_mols.txt> [sanitize:0|1, default 1]
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

#include <GraphMol/GraphMol.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/SmilesParse/SmilesParse.h>

int main(int argc, char** argv) {
  bool sanitize = argc > 2 ? (std::atoi(argv[2]) != 0) : true;
  std::ifstream in(argv[1]);
  std::string line;
  long long idx = 0;
  while (std::getline(in, line)) {
    std::fprintf(stderr, "about to parse %lld\n", idx);
    std::fflush(stderr);
    if (!line.empty() && line.back() == '\r') line.pop_back();
    size_t t = line.find('\t');
    std::string smi = t == std::string::npos ? line : line.substr(t + 1);
    std::unique_ptr<RDKit::RWMol> mol(RDKit::SmilesToMol(smi, 0, false));
    std::fprintf(stderr, "  unsanitized parse %lld -> %s\n", idx, mol ? "ok" : "null");
    std::fflush(stderr);
    if (mol && sanitize) {
      std::fprintf(stderr, "  about to sanitize %lld\n", idx);
      std::fflush(stderr);
      try {
        RDKit::MolOps::sanitizeMol(*mol);
        std::fprintf(stderr, "  sanitize %lld -> ok\n", idx);
      } catch (const std::exception& e) {
        std::fprintf(stderr, "  sanitize %lld -> THREW: %s\n", idx, e.what());
      } catch (...) {
        std::fprintf(stderr, "  sanitize %lld -> THREW unknown\n", idx);
      }
      std::fflush(stderr);
    }
    ++idx;
  }
  std::fprintf(stderr, "done, %lld lines\n", idx);
  return 0;
}
