// leak_check: is a holdout set clean against a training library beyond the
// molecule-key exclusion? (cnn_STATUS.md, after cnn_a's large hxqry gain)
//  (1) nearest-neighbour Tanimoto (fingerprint bits <tag>) of each query's truth
//      to every library molecule -> distribution (analog density)
//  (2) duplicated spectra: query spectra whose (precursor, peak list) also occur
//      in the library under another molecule
//   leak_check <dir> <qry> <win stem> <lib> <tag> [threads]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "parallel.hpp"

struct Head { float precursor; int32_t mol, off, n, tag; };

static std::vector<char> slurp(const std::string& p) {
  std::ifstream f(p, std::ios::binary | std::ios::ate);
  if (!f) { std::fprintf(stderr, "cannot read %s\n", p.c_str()); std::exit(1); }
  std::streamsize n = f.tellg(); f.seekg(0);
  std::vector<char> b((size_t(n))); f.read(b.data(), n);
  return b;
}
struct Fps { std::vector<uint64_t> ptr; std::vector<uint16_t> bits; uint32_t nBits = 0; };
static Fps loadFps(const std::string& path) {
  Fps f; auto b = slurp(path); const char* p = b.data();
  uint32_t n; std::memcpy(&n, p, 4); std::memcpy(&f.nBits, p + 4, 4); p += 8;
  f.ptr.resize(n + 1); f.ptr[0] = 0;
  for (uint32_t i = 0; i < n; ++i) {
    uint16_t c; std::memcpy(&c, p, 2); p += 2;
    size_t o = f.bits.size(); f.bits.resize(o + c); std::memcpy(f.bits.data() + o, p, 2 * size_t(c)); p += 2 * size_t(c);
    f.ptr[i + 1] = f.bits.size();
  }
  return f;
}
struct Spec { std::vector<Head> h; std::vector<float> data; };
static Spec loadSpec(const std::string& dir, const std::string& stem) {
  Spec s; auto hb = slurp(dir + "/" + stem + "_head.bin");
  s.h.resize(hb.size() / sizeof(Head)); std::memcpy(s.h.data(), hb.data(), hb.size());
  auto pb = slurp(dir + "/" + stem + "_peaks.bin");
  s.data.resize(pb.size() / 4); std::memcpy(s.data.data(), pb.data(), pb.size());
  size_t cur = 0;
  for (auto& h : s.h) { h.off = int32_t(cur); cur += 2 * size_t(h.n); }  // mz[n] then int[n]
  return s;
}
static uint64_t specKey(const Head& h, const float* mz) {
  uint64_t k = uint64_t(std::llround(h.precursor * 100.0)) * 1000003ull + uint64_t(h.n);
  std::vector<int64_t> m(h.n);
  for (int i = 0; i < h.n; ++i) m[i] = std::llround(mz[i] * 100.0);
  std::sort(m.begin(), m.end());
  for (int64_t x : m) k = (k ^ uint64_t(x)) * 1099511628211ull;
  return k;
}

int main(int argc, char** argv) {
  if (argc < 6) { std::fprintf(stderr, "usage: leak_check dir qry win lib tag [threads]\n"); return 1; }
  std::string dir = argv[1], qry = argv[2], win = argv[3], lib = argv[4], tag = argv[5];
  parallel_cap() = argc > 6 ? unsigned(std::atoi(argv[6])) : 5u;
  // (1) truth fingerprints from the window cache
  Fps W = loadFps(dir + "/" + win + "_" + tag + ".bin");
  Fps L = loadFps(dir + "/" + lib + "_" + tag + ".bin");
  auto wb = slurp(dir + "/" + win + "_win.bin");
  const char* p = wb.data(); uint32_t nq; std::memcpy(&nq, p, 4); p += 4;
  std::vector<uint32_t> truthId;
  for (uint32_t q = 0; q < nq; ++q) {
    int32_t mol, tr; uint32_t n;
    std::memcpy(&mol, p, 4); std::memcpy(&tr, p + 4, 4); std::memcpy(&n, p + 8, 4); p += 12;
    const uint32_t* ids = reinterpret_cast<const uint32_t*>(p); p += 4 * size_t(n);
    if (tr >= 0) truthId.push_back(ids[tr]);
  }
  const size_t nL = L.ptr.size() - 1;
  std::vector<double> best(truthId.size());
  ::parallel_for(int64_t(truthId.size()), [&](int64_t i) {
    std::vector<uint8_t> on(W.nBits, 0);
    uint32_t id = truthId[i]; int na = int(W.ptr[id + 1] - W.ptr[id]);
    for (uint64_t k = W.ptr[id]; k < W.ptr[id + 1]; ++k) on[W.bits[k]] = 1;
    double bt = 0;
    for (size_t m = 0; m < nL; ++m) {
      int nb = int(L.ptr[m + 1] - L.ptr[m]), in = 0;
      for (uint64_t k = L.ptr[m]; k < L.ptr[m + 1]; ++k) in += on[L.bits[k]];
      double t = double(in) / std::max(1, na + nb - in);
      bt = std::max(bt, t);
    }
    best[i] = bt;
  });
  std::sort(best.begin(), best.end());
  const double TH[] = {1.0, 0.95, 0.9, 0.8, 0.7, 0.6};
  std::printf("%s vs %s (%s bits): %zu truths in window; nearest-neighbour Tanimoto median %.3f, mean %.3f\n", qry.c_str(),
              lib.c_str(), tag.c_str(), best.size(), best[best.size() / 2],
              [&] { double s = 0; for (double x : best) s += x; return s / best.size(); }());
  for (double th : TH) {
    size_t c = 0; for (double x : best) c += x >= th - 1e-12;
    std::printf("  NN Tanimoto >= %.2f : %5.1f %%\n", th, 100.0 * c / best.size());
  }
  // (2) duplicate spectra
  Spec Q = loadSpec(dir, qry), B = loadSpec(dir, lib);
  std::unordered_map<uint64_t, int> libKeys;
  for (auto& h : B.h) if (h.n > 0) libKeys[specKey(h, B.data.data() + h.off)] = h.mol;
  int dup = 0, tot = 0;
  for (auto& h : Q.h) { if (h.n <= 0) continue; ++tot; dup += libKeys.count(specKey(h, Q.data.data() + h.off)) > 0; }
  std::printf("  query spectra with an identical (precursor, m/z list @0.01) spectrum in %s: %d / %d\n", lib.c_str(), dup, tot);
  return 0;
}
