// Narrow PubChem's CID-Mass to just the 400 real test molecules' own mass
// windows, instead of one broad range. The broad [245, 461] Da range alone
// kept 75.8M of PubChem's 124.7M CIDs (61%) - PubChem's compound mass
// distribution is heavily concentrated in exactly the drug-like range this
// competition's test set lives in, so a blanket range is not a filter at
// PubChem's scale. What actually narrows it is the same thing match.cpp's
// own candidate gate always was: a tight window around each molecule's own
// (adduct-corrected) neutral mass, not the test set's overall span.
//
// Usage:
//   pubchem_target_filter <test_head.bin> <CID-Mass.tsv or pre-filtered.tsv> <ppm> <out.tsv>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <string>
#include <vector>

namespace {

struct Head { float precursor; int32_t mol, off, n, tag; };

constexpr float ADDUCT_SHIFT[10] = {
    1.007276f, -1.007276f, 22.989218f, 18.033823f, 38.963158f,
    34.969402f, 44.998201f, -17.002740f, 1.007276f, 22.989218f,
};

}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr, "usage: %s <test_head.bin> <CID-Mass.tsv> <ppm> <out.tsv>\n", argv[0]);
    return 1;
  }
  float ppm = float(std::atof(argv[3]));

  std::ifstream hf(argv[1], std::ios::binary | std::ios::ate);
  if (!hf) { std::fprintf(stderr, "cannot read %s\n", argv[1]); return 1; }
  size_t n = size_t(hf.tellg()) / sizeof(Head);
  hf.seekg(0);
  std::vector<Head> heads(n);
  hf.read(reinterpret_cast<char*>(heads.data()), n * sizeof(Head));

  std::set<float> targetSet;
  for (const Head& h : heads) {
    int adductIdx = h.tag < 0 ? -1 : h.tag / 16;
    float shift = (adductIdx >= 0 && adductIdx < 10) ? ADDUCT_SHIFT[adductIdx] : 1.007276f;
    targetSet.insert(h.precursor - shift);
  }
  std::vector<float> targets(targetSet.begin(), targetSet.end());
  std::fprintf(stderr, "%zu spectra -> %zu distinct target neutral masses, ppm=%.1f\n",
               n, targets.size(), ppm);

  std::ifstream in(argv[2]);
  if (!in) { std::fprintf(stderr, "cannot read %s\n", argv[2]); return 1; }
  std::ofstream out(argv[4]);

  std::string line;
  long long total = 0, kept = 0;
  while (std::getline(in, line)) {
    ++total;
    size_t t1 = line.find('\t');
    if (t1 == std::string::npos) continue;
    // the broad-filter output (pubchem_filter's) is already CID\tmass (2
    // columns); the raw CID-Mass.tsv is CID\tformula\tmass\tmass (4) - handle
    // both by taking the LAST tab-separated field as the mass.
    size_t lastTab = line.rfind('\t');
    double mass = std::atof(line.c_str() + lastTab + 1);
    float window = float(mass) * ppm * 1e-6f;
    auto it = std::lower_bound(targets.begin(), targets.end(), float(mass) - window);
    bool hit = false;
    while (it != targets.end() && *it <= float(mass) + window) {
      if (std::fabs(*it - float(mass)) <= window) { hit = true; break; }
      ++it;
    }
    if (hit) {
      long long cid = std::atoll(line.c_str());
      out << cid << '\t' << mass << '\n';
      ++kept;
    }
    if (total % 20000000 == 0)
      std::fprintf(stderr, "  %lld lines, %lld kept\n", total, kept);
  }
  std::fprintf(stderr, "done: %lld lines read, %lld within %.1f ppm of a target mass\n",
               total, kept, ppm);
  return 0;
}
