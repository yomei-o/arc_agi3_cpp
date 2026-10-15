// CPU inference timing of a TFM1 model (tfm_model.hpp, float CPU build) on a
// query set, as phys_gentfm runs it (batches of 64 spectra ordered by peak count).
//   tfm_cputime <dir> <qry stem> <model.bin> <threads> [max spectra]
// Build like Kaggle: g++ -O3 -std=c++17 -pthread tfm_cputime.cpp (optionally -march=x86-64-v3)
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "parallel.hpp"
#include "tfm_model.hpp"

struct Head { float precursor; int32_t mol, off, n, tag; };
constexpr float MODE[10] = {1, -1, 1, 1, 1, -1, -1, 1, 1, 1};

static std::vector<char> slurp(const std::string& p) {
  std::ifstream f(p, std::ios::binary | std::ios::ate);
  if (!f) { std::fprintf(stderr, "cannot read %s\n", p.c_str()); std::exit(1); }
  std::streamsize n = f.tellg(); f.seekg(0);
  std::vector<char> b((size_t(n))); f.read(b.data(), n);
  return b;
}

int main(int argc, char** argv) {
  if (argc < 5) { std::fprintf(stderr, "usage: tfm_cputime <dir> <qry> <model.bin> <threads> [max]\n"); return 1; }
  std::string dir = argv[1], qs = argv[2];
  parallel_cap() = unsigned(std::atoi(argv[4]));
  size_t maxS = argc > 5 ? size_t(std::atol(argv[5])) : size_t(-1);
  std::vector<char> hb = slurp(dir + "/" + qs + "_head.bin");
  std::vector<Head> head(hb.size() / sizeof(Head));
  std::memcpy(head.data(), hb.data(), hb.size());
  std::vector<char> pb = slurp(dir + "/" + qs + "_peaks.bin");
  std::vector<float> mz, it;
  const float* p = reinterpret_cast<const float*>(pb.data());
  for (Head& h : head) {
    h.off = int32_t(mz.size());
    mz.insert(mz.end(), p, p + h.n); p += h.n;
    it.insert(it.end(), p, p + h.n); p += h.n;
  }
  if (head.size() > maxS) head.resize(maxS);
  tfm::Model m;
  if (!m.load(dir + "/" + argv[3], false)) { std::fprintf(stderr, "cannot load model\n"); return 1; }
  auto t0 = std::chrono::steady_clock::now();
  const int maxP = m.c.maxPeaks;
  std::vector<std::vector<float>> M(head.size()), I(head.size());
  for (size_t s = 0; s < head.size(); ++s) {
    const Head& h = head[s];
    const float* x = mz.data() + h.off; const float* y = it.data() + h.off;
    float mx = 0.f;
    for (int i = 0; i < h.n; ++i) if (x[i] > 0.f && x[i] <= h.precursor + 2.f) mx = std::max(mx, y[i]);
    std::vector<int> idx;
    if (mx > 0.f)
      for (int i = 0; i < h.n; ++i) if (x[i] > 0.f && x[i] <= h.precursor + 2.f && y[i] / mx >= 1e-3f) idx.push_back(i);
    if (int(idx.size()) > maxP) {
      std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return y[a] > y[b]; });
      idx.resize(maxP); std::sort(idx.begin(), idx.end());
    }
    for (int i : idx) { M[s].push_back(x[i]); I[s].push_back(std::sqrt(y[i] / mx)); }
    if (M[s].empty()) { M[s].push_back(0.f); I[s].push_back(0.f); }
  }
  std::vector<int> ord(head.size());
  for (size_t i = 0; i < ord.size(); ++i) ord[i] = int(i);
  std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return M[a].size() < M[b].size(); });
  tfm::HostBatch b;
  std::vector<float> z;
  double chk = 0; long long tok = 0;
  for (size_t s0 = 0; s0 < ord.size(); s0 += 64) {
    size_t s1 = std::min(ord.size(), s0 + 64);
    int B = int(s1 - s0), mx = 1;
    for (size_t i = s0; i < s1; ++i) mx = std::max(mx, int(M[ord[i]].size()));
    b.B = B; b.N = 1 + mx;
    b.mz.assign(size_t(B) * mx, 0.f); b.it.assign(size_t(B) * mx, 0.f);
    b.prec.resize(B); b.mode.resize(B); b.np.resize(B); b.ad.resize(B); b.in.resize(B);
    for (int k = 0; k < B; ++k) {
      int i = ord[s0 + k];
      const Head& h = head[i];
      std::copy(M[i].begin(), M[i].end(), b.mz.begin() + size_t(k) * mx);
      std::copy(I[i].begin(), I[i].end(), b.it.begin() + size_t(k) * mx);
      b.np[k] = int(M[i].size()); b.prec[k] = h.precursor; tok += 1 + b.np[k];
      int tag = h.tag, a = -1, si = -1;
      if (tag >= 0) { bool u = tag % 16 == 15; a = u ? tag / 16 + 1 : tag / 16; si = u ? -1 : tag % 16; }
      b.ad[k] = (a >= 0 && a < 10) ? a : 10; b.in[k] = (si >= 0 && si < 5) ? si : 5;
      b.mode[k] = (a >= 0 && a < 10) ? MODE[a] : 0.f;
    }
    m.setBatch(b);
    m.forward(false, 0.f, 0);
    z.resize(size_t(B) * m.c.NB);
    tfm::download(m.Z, z.data(), int64_t(z.size()));
    for (float v : z) chk += v;
  }
  double tt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("%s d%d L%d: %zu spectra, mean %.1f tokens, %.2f s with %s threads = %.2f ms/spectrum (checksum %.6g)\n", argv[3],
              m.c.d, m.c.L, head.size(), double(tok) / head.size(), tt, argv[4], 1000.0 * tt / head.size(), chk);
  return 0;
}
