// Build a mass-sorted, whole-PubChem candidate table for runtime retrieval.
//
// The earlier PubChem "expansion" (pubchem_target_filter.cpp -> pool) kept
// only CIDs within 5 ppm of the VISIBLE test set's masses - but the visible
// test.parquet is a placeholder drawn from train; the scoring rerun swaps in
// a hidden test set (the top public notebooks all say so explicitly). So
// that filter was keyed to the wrong molecules and is close to useless at
// scoring time. This instead keeps every single-component PubChem compound
// in a broad mass range, sorted by monoisotopic mass, so the kernel can pull
// a +-ppm window around whatever precursor the hidden test actually has.
//
// Inputs: CID-Mass.tsv (CID, formula, monoisotopic mass, exact mass) and
// CID-SMILES.tsv (CID, SMILES), both from PubChem's Compound/Extras.
// Outputs (little-endian):
//   pc_mass.bin  float32[N], ascending
//   pc_cid.bin   int32[N]
//   pc_off.bin   uint64[N+1], byte offsets into pc_smi.bin
//   pc_smi.bin   concatenated SMILES (no separators)
//
// Usage: pc_build <dir> <mass_lo> <mass_hi> <max_smiles_len>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  if (argc < 5) { std::fprintf(stderr, "usage: %s <dir> <lo> <hi> <maxlen>\n", argv[0]); return 1; }
  std::string dir = argv[1];
  double lo = std::atof(argv[2]), hi = std::atof(argv[3]);
  size_t maxLen = size_t(std::atoi(argv[4]));

  std::vector<float> massOfCid;
  {
    std::ifstream f(dir + "/CID-Mass.tsv");
    if (!f) { std::fprintf(stderr, "cannot read CID-Mass.tsv\n"); return 1; }
    std::string line;
    long long n = 0;
    while (std::getline(f, line)) {
      size_t t1 = line.find('\t');
      if (t1 == std::string::npos) continue;
      size_t t2 = line.find('\t', t1 + 1);
      if (t2 == std::string::npos) continue;
      long long cid = std::atoll(line.c_str());
      if (cid <= 0) continue;
      if (size_t(cid) >= massOfCid.size()) massOfCid.resize(size_t(cid) + 1 + (size_t(cid) >> 4), 0.f);
      massOfCid[size_t(cid)] = float(std::atof(line.c_str() + t2 + 1));
      if (++n % 20000000 == 0) std::fprintf(stderr, "  mass %lld\n", n);
    }
    std::fprintf(stderr, "CID-Mass: %lld rows, max cid %zu\n", n, massOfCid.size());
  }

  std::string smiBuf;
  smiBuf.reserve(size_t(6) << 30);
  std::vector<float> mass;
  std::vector<int32_t> cids;
  std::vector<uint64_t> off;
  std::vector<uint32_t> len;
  {
    std::ifstream f(dir + "/CID-SMILES.tsv");
    if (!f) { std::fprintf(stderr, "cannot read CID-SMILES.tsv\n"); return 1; }
    std::string line;
    long long n = 0, kept = 0, mixtures = 0, outOfRange = 0, tooLong = 0;
    while (std::getline(f, line)) {
      ++n;
      size_t t = line.find('\t');
      if (t == std::string::npos) continue;
      long long cid = std::atoll(line.c_str());
      if (cid <= 0 || size_t(cid) >= massOfCid.size()) continue;
      float m = massOfCid[size_t(cid)];
      if (m < lo || m > hi) { ++outOfRange; continue; }
      const char* s = line.c_str() + t + 1;
      size_t l = line.size() - t - 1;
      if (l == 0) continue;
      if (std::memchr(s, '.', l)) { ++mixtures; continue; }
      if (l > maxLen) { ++tooLong; continue; }
      mass.push_back(m);
      cids.push_back(int32_t(cid));
      off.push_back(smiBuf.size());
      len.push_back(uint32_t(l));
      smiBuf.append(s, l);
      ++kept;
      if (n % 20000000 == 0) std::fprintf(stderr, "  smiles %lld read, %lld kept\n", n, kept);
    }
    std::fprintf(stderr, "CID-SMILES: %lld rows -> %lld kept (%lld mixtures, %lld out of [%.0f,%.0f], %lld longer than %zu)\n",
                 n, kept, mixtures, outOfRange, lo, hi, tooLong, maxLen);
  }

  std::vector<uint32_t> order(mass.size());
  for (size_t i = 0; i < order.size(); ++i) order[i] = uint32_t(i);
  std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
    return mass[a] != mass[b] ? mass[a] < mass[b] : cids[a] < cids[b];
  });
  std::fprintf(stderr, "sorted %zu\n", order.size());

  std::ofstream fm(dir + "/pc_mass.bin", std::ios::binary), fc(dir + "/pc_cid.bin", std::ios::binary);
  std::ofstream fo(dir + "/pc_off.bin", std::ios::binary), fs(dir + "/pc_smi.bin", std::ios::binary);
  uint64_t pos = 0;
  for (uint32_t i : order) {
    fm.write(reinterpret_cast<const char*>(&mass[i]), 4);
    fc.write(reinterpret_cast<const char*>(&cids[i]), 4);
    fo.write(reinterpret_cast<const char*>(&pos), 8);
    fs.write(smiBuf.data() + off[i], len[i]);
    pos += len[i];
  }
  fo.write(reinterpret_cast<const char*>(&pos), 8);
  std::fprintf(stderr, "wrote pc_*.bin: %zu compounds, %.2f GB of SMILES\n", order.size(), double(pos) / 1e9);
  return 0;
}
