// Stage 3 of PubChem-scale pool expansion: parse the mass-filtered PubChem
// candidates with RDKit and fold them into the existing train+COCONUT pool.
//
// Dedup key is RDKit's own canonical SMILES, NOT InChIKey14 as build_pool.py
// used for the COCONUT merge. The minimal RDKit extraction in this directory
// never pulled in the INCHI-API wrapper module (it links against the
// external IUPAC InChI C library, a separate codebase this extraction never
// touched), so InChIKey generation is not available here. Canonical SMILES
// catches the common case this merge actually faces - the exact same PubChem
// CID also present (reformatted) in COCONUT or train - but unlike InChIKey14
// it is NOT stereochemistry-blind and NOT representation-invariant across
// RDKit versions/options the way InChI is designed to be. This is a weaker
// de-dup than the rest of the pool uses, noted here rather than silently
// assumed equivalent.
//
// Mass is recomputed the same way forward_model.h's fragmentMass does (sum
// of most-common-isotope atomic masses + implicit H), not via a Descriptors
// module call - Descriptors was never pulled into this extraction either.
//
// Usage:
//   pubchem_parse <existing pool_mols.txt> <candidates.tsv> <out_new.tsv>
// out_new.tsv columns: name(=PUBCHEM:<cid>) \t smiles \t mass, only for
// candidates whose canonical SMILES was not already in the existing pool.
#include <atomic>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include <GraphMol/GraphMol.h>
#include <GraphMol/PeriodicTable.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/SmilesParse/SmilesWrite.h>

namespace {

double molMass(const RDKit::ROMol& mol) {
  const RDKit::PeriodicTable* pt = RDKit::PeriodicTable::getTable();
  double mass = 0.0;
  for (const auto& a : mol.atoms()) {
    mass += pt->getMostCommonIsotopeMass(a->getAtomicNum());
    mass += double(a->getTotalNumHs(false)) * pt->getMostCommonIsotopeMass(1);
  }
  return mass;
}

std::string canonicalSmiles(const RDKit::ROMol& mol) {
  return RDKit::MolToSmiles(mol);
}

// RDKit::SmilesToMol sanitizes by default, and sanitization legitimately
// throws (RDKit::MolSanitizeException and friends) on structures with
// invalid valence etc - confirmed on a real pool entry, a polyphosphate
// written with plain "=O" bonds giving one P a valence of 6
// ("Explicit valence for atom # 1 P, 6, is greater than permitted"). An
// uncaught exception terminates the whole batch job, not just that one
// molecule, so every parse in this file goes through here instead of a bare
// SmilesToMol call.
RDKit::RWMol* safeParse(const std::string& smi) {
  try {
    return RDKit::SmilesToMol(smi);
  } catch (...) {
    return nullptr;
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: %s <pool_mols.txt> <candidates.tsv> <out_new.tsv>\n", argv[0]);
    return 1;
  }
  RDKit::PeriodicTable::getTable();  // single-threaded init

  std::unordered_set<std::string> seen;
  {
    std::ifstream in(argv[1]);
    if (!in) { std::fprintf(stderr, "cannot read %s\n", argv[1]); return 1; }
    std::string line;
    long long total = 0, parsed = 0;
    while (std::getline(in, line)) {
      if (!line.empty() && line.back() == '\r') line.pop_back();
      size_t t = line.find('\t');
      if (t == std::string::npos) continue;
      std::string smi = line.substr(t + 1);
      ++total;
      std::unique_ptr<RDKit::RWMol> mol(safeParse(smi));
      if (!mol) continue;
      seen.insert(canonicalSmiles(*mol));
      ++parsed;
      if (total % 100000 == 0)
        std::fprintf(stderr, "  existing pool: %lld/%lld parsed\n", parsed, total);
    }
    std::fprintf(stderr, "existing pool: %zu canonical SMILES loaded (%lld/%lld parsed)\n",
                 seen.size(), parsed, total);
  }

  // Read all candidate rows up front (cid, mass-string unused, smiles) so
  // the parse loop below can be split across threads by index.
  std::vector<long long> cids;
  std::vector<std::string> smiles;
  {
    std::ifstream in(argv[2]);
    if (!in) { std::fprintf(stderr, "cannot read %s\n", argv[2]); return 1; }
    std::string line;
    std::getline(in, line);  // header
    while (std::getline(in, line)) {
      if (!line.empty() && line.back() == '\r') line.pop_back();
      size_t t1 = line.find('\t');
      if (t1 == std::string::npos) continue;
      size_t t2 = line.find('\t', t1 + 1);
      if (t2 == std::string::npos) continue;
      cids.push_back(std::atoll(line.c_str()));
      smiles.push_back(line.substr(t2 + 1));
    }
  }
  std::fprintf(stderr, "candidates: %zu rows to parse\n", cids.size());

  std::vector<std::string> outLine(cids.size());
  std::atomic<size_t> next(0);
  std::atomic<long long> nParsed(0), nFailed(0), nDup(0), nKept(0);
  std::mutex seenMutex;
  std::mutex printMutex;

  unsigned nthread = std::max(1u, std::thread::hardware_concurrency());
  std::vector<std::thread> pool;
  for (unsigned t = 0; t < nthread; ++t) {
    pool.emplace_back([&]() {
      for (;;) {
        size_t i = next++;
        if (i >= cids.size()) break;
        std::unique_ptr<RDKit::RWMol> mol(safeParse(smiles[i]));
        if (!mol) { ++nFailed; continue; }
        ++nParsed;
        std::string canon = canonicalSmiles(*mol);
        {
          std::lock_guard<std::mutex> lk(seenMutex);
          if (seen.count(canon)) { ++nDup; continue; }
          seen.insert(canon);
        }
        double mass = molMass(*mol);
        std::ostringstream row;
        row << "PUBCHEM:" << cids[i] << '\t' << canon << '\t' << mass;
        outLine[i] = row.str();
        ++nKept;
        long long done = nParsed.load();
        if (done % 200000 == 0) {
          std::lock_guard<std::mutex> lk(printMutex);
          std::fprintf(stderr, "  parsed %lld, failed %lld, dup %lld, kept %lld\n",
                       done, nFailed.load(), nDup.load(), nKept.load());
        }
      }
    });
  }
  for (auto& th : pool) th.join();

  std::ofstream out(argv[3]);
  long long written = 0;
  for (auto& l : outLine) {
    if (l.empty()) continue;
    out << l << '\n';
    ++written;
  }
  std::fprintf(stderr,
               "done: %zu candidates, %lld parsed, %lld failed, %lld dup-vs-pool, %lld new structures written\n",
               cids.size(), nParsed.load(), nFailed.load(), nDup.load(), written);
  return 0;
}
