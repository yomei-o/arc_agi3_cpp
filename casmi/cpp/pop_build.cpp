// Popularity prior per PubChem compound, aligned with pc_build.cpp's sorted
// table: log1p(#substances depositing the CID) + log1p(#PubMed refs). The
// top public notebooks add 0.15-0.25 x this to their ranker score: of two
// isomers that fit a spectrum equally well, the one more chemists have
// registered and written about is far more likely to be what was measured.
//
// Reads the decompressed CID-SID and CID-PMID files (first column = CID,
// one row per (CID, SID/PMID) pair) and pc_cid.bin; writes pc_pop.bin.
// Usage: pop_build <dir> <CID-SID.tsv> <CID-PMID.tsv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

static void countFirstColumn(const std::string& path, std::vector<uint32_t>& cnt) {
  std::ifstream f(path);
  if (!f) { std::fprintf(stderr, "cannot read %s\n", path.c_str()); std::exit(1); }
  std::string line;
  long long n = 0;
  while (std::getline(f, line)) {
    long long cid = std::atoll(line.c_str());
    if (cid <= 0) continue;
    if (size_t(cid) >= cnt.size()) cnt.resize(size_t(cid) + 1 + (size_t(cid) >> 4), 0);
    ++cnt[size_t(cid)];
    if (++n % 50000000 == 0) std::fprintf(stderr, "  %s: %lld\n", path.c_str(), n);
  }
  std::fprintf(stderr, "%s: %lld rows\n", path.c_str(), n);
}

int main(int argc, char** argv) {
  if (argc < 4) { std::fprintf(stderr, "usage: %s <dir> <CID-SID> <CID-PMID>\n", argv[0]); return 1; }
  std::string dir = argv[1];
  std::vector<uint32_t> sid, pmid;
  countFirstColumn(argv[2], sid);
  countFirstColumn(argv[3], pmid);

  std::ifstream fc(dir + "/pc_cid.bin", std::ios::binary | std::ios::ate);
  size_t n = size_t(fc.tellg()) / 4;
  fc.seekg(0);
  std::vector<int32_t> cid(n);
  fc.read(reinterpret_cast<char*>(cid.data()), n * 4);
  std::vector<float> pop(n);
  double mean = 0;
  size_t withRefs = 0;
  for (size_t i = 0; i < n; ++i) {
    size_t c = size_t(cid[i]);
    uint32_t s = c < sid.size() ? sid[c] : 0, p = c < pmid.size() ? pmid[c] : 0;
    pop[i] = float(std::log1p(double(s)) + std::log1p(double(p)));
    mean += pop[i];
    withRefs += p > 0;
  }
  std::ofstream o(dir + "/pc_pop.bin", std::ios::binary);
  o.write(reinterpret_cast<const char*>(pop.data()), n * 4);
  std::fprintf(stderr, "wrote pc_pop.bin: %zu compounds, mean popularity %.3f, %zu with PubMed refs\n",
               n, mean / double(n), withRefs);
  return 0;
}
