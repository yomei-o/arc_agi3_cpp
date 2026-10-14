// Stage 1 of PubChem-scale candidate retrieval: filter CID-Mass.gz by mass
// range, no RDKit needed yet. PubChem has ~1.1e8 compounds; parsing every
// one of them with RDKit just to find out most are irrelevant would waste
// the bulk of the time. CID-Mass.gz carries PubChem's own precomputed
// monoisotopic mass per CID, so this step is pure text/number filtering -
// the expensive RDKit step only ever touches the CIDs that survive here.
//
// Input is the gunzipped CID-Mass file - NOT the 2-column "CID<TAB>mass"
// this was first written assuming. The real format, confirmed by looking
// at the actual decompressed file, is 4 tab-separated columns:
//   CID  formula  mass  mass
// (the two mass columns read identical in every row checked - possibly
// monoisotopic mass repeated, or mass and monoisotopic mass coinciding for
// non-isotope-labelled entries; either way column 3 is used here).
// Output is just the surviving CIDs plus their mass, one per line, for the
// next stage (pubchem_join) to pick out of CID-SMILES by CID.
//
// Usage:
//   pubchem_filter <CID-Mass (gunzipped)> <mass_lo> <mass_hi> <out.tsv>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr, "usage: %s <CID-Mass.tsv> <mass_lo> <mass_hi> <out.tsv>\n", argv[0]);
    return 1;
  }
  double lo = std::atof(argv[2]), hi = std::atof(argv[3]);
  std::ifstream in(argv[1]);
  if (!in) { std::fprintf(stderr, "cannot read %s\n", argv[1]); return 1; }
  std::ofstream out(argv[4]);

  std::string line;
  long long total = 0, kept = 0;
  while (std::getline(in, line)) {
    ++total;
    // column 0: CID, column 1: formula, column 2: mass (the one we want)
    size_t t1 = line.find('\t');
    if (t1 == std::string::npos) continue;
    size_t t2 = line.find('\t', t1 + 1);
    if (t2 == std::string::npos) continue;
    size_t t3 = line.find('\t', t2 + 1);
    std::string massStr = line.substr(t2 + 1, t3 == std::string::npos ? std::string::npos : t3 - t2 - 1);
    double mass = std::atof(massStr.c_str());
    if (mass >= lo && mass <= hi) {
      long long cid = std::atoll(line.c_str());
      out << cid << '\t' << massStr << '\n';
      ++kept;
    }
    if (total % 20000000 == 0)
      std::fprintf(stderr, "  %lld lines, %lld kept\n", total, kept);
  }
  std::fprintf(stderr, "done: %lld lines read, %lld in [%.2f, %.2f]\n", total, kept, lo, hi);
  return 0;
}
