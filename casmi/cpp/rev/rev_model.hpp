// Reverse model: MS/MS spectrum -> ECFP on-bit probabilities. The second
// scoring signal Sho Saga's writeup measures as complementary to the
// forward model (rank correlation 0.37 between the two; reverse alone 0.474,
// forward alone 0.640, combined 0.674 on their validation set) and the one
// this project had no version of at all.
//
// A plain MLP over a sparse binned spectrum, with every gradient written by
// hand rather than taped: the architecture is fixed, so a hand-derived
// backward is short and lets the first layer stay a sparse-times-dense
// product in both directions (CSR forward, CSC for dW1 - one owner per
// output element either way, so no atomics on the GPU and bit-identical
// results on the CPU path). All device work goes through backend.hpp
// (vendored from yomei-o/yolov8_cpp): the same source builds with g++/MSVC
// for the CPU and with nvcc -DUSE_CUDA -DUSE_CUBLAS for the RTX 3090.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "backend.hpp"

namespace rev {

// ---- input layout ----------------------------------------------------------
constexpr float BIN = 0.1f;
constexpr int NB_FRAG = 10000;      // fragment m/z 0..1000 Da
constexpr int NB_NL = 3000;         // neutral loss 0..300 Da
constexpr int N_ADDUCT = 10, N_INSTR = 5;
constexpr int OFF_NL = NB_FRAG, OFF_ADDUCT = NB_FRAG + NB_NL;
constexpr int OFF_INSTR = OFF_ADDUCT + N_ADDUCT, OFF_MASS = OFF_INSTR + N_INSTR;
constexpr int D_IN = OFF_MASS + 1;
constexpr int MAX_PEAKS = 300;      // keep the most intense; bounds nnz per row

struct Head { float precursor; int32_t mol, off, n, tag; };

constexpr float ADDUCT_SHIFT[10] = {
    1.007276f, -1.007276f, 22.989218f, 18.033823f, 38.963158f,
    34.969402f, 44.998201f, -17.002740f, 1.007276f, 22.989218f,
};

// Sparse row for one spectrum: sorted unique (column, value).
inline void featurize(const Head& h, const float* mz, const float* it,
                      std::vector<std::pair<int, float>>& row) {
  row.clear();
  int n = h.n;
  std::vector<int> idx(n);
  for (int i = 0; i < n; ++i) idx[i] = i;
  if (n > MAX_PEAKS) {
    std::nth_element(idx.begin(), idx.begin() + MAX_PEAKS, idx.end(),
                     [&](int a, int b) { return it[a] > it[b]; });
    idx.resize(MAX_PEAKS);
  }
  float mx = 0.f;
  for (int i : idx) mx = std::max(mx, it[i]);
  if (mx <= 0.f) mx = 1.f;
  for (int i : idx) {
    float v = std::sqrt(std::max(0.f, it[i]) / mx);
    int fb = int(mz[i] / BIN);
    if (fb >= 0 && fb < NB_FRAG) row.emplace_back(fb, v);
    float nl = h.precursor - mz[i];
    int nb = int(nl / BIN);
    if (nl > 0.f && nb >= 0 && nb < NB_NL) row.emplace_back(OFF_NL + nb, v);
  }
  // tag = adduct*16 + instrument with -1 for an unknown instrument, which
  // borrows from the adduct: 16a-1 reads as adduct a-1, instrument 15.
  int a = -1, s = -1;
  if (h.tag >= 0) {
    bool unkInstr = h.tag % 16 == 15;
    a = unkInstr ? h.tag / 16 + 1 : h.tag / 16;
    s = unkInstr ? -1 : h.tag % 16;
  }
  if (a >= 0 && a < N_ADDUCT) row.emplace_back(OFF_ADDUCT + a, 1.f);
  if (s >= 0 && s < N_INSTR) row.emplace_back(OFF_INSTR + s, 1.f);
  row.emplace_back(OFF_MASS, h.precursor / 1000.f);
  std::sort(row.begin(), row.end());
  size_t w = 0;
  for (size_t r = 0; r < row.size(); ++r) {
    if (w > 0 && row[w - 1].first == row[r].first) row[w - 1].second = std::max(row[w - 1].second, row[r].second);
    else row[w++] = row[r];
  }
  row.resize(w);
}

// ---- device helpers --------------------------------------------------------
template <class T>
inline void upload(bk::Buffer<T>& b, const std::vector<T>& h) {
  if (b.size() < int64_t(h.size()) || !b.data()) b.alloc(std::max<int64_t>(1, int64_t(h.size())));
#ifdef USE_CUDA
  if (!h.empty()) BK_CHECK(cudaMemcpy(b.data(), h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
#else
  if (!h.empty()) std::memcpy(b.data(), h.data(), h.size() * sizeof(T));
#endif
}
template <class T>
inline void download(const bk::Buffer<T>& b, std::vector<T>& h, int64_t n) {
  h.resize(n);
#ifdef USE_CUDA
  BK_CHECK(cudaMemcpy(h.data(), b.data(), n * sizeof(T), cudaMemcpyDeviceToHost));
#else
  std::memcpy(h.data(), b.data(), n * sizeof(T));
#endif
}
inline void fill(bk::Buffer<float>& b, int64_t n, float v) {
  float* p = b.data();
  bk::parallel_for(n, [=] BK_HD (int64_t i) { p[i] = v; });
}

struct Param {
  int64_t n = 0;
  bk::Buffer<float> w, g, m, v;
  void init(int64_t count, const std::vector<float>& host) {
    n = count;
    upload(w, host);
    g.alloc(n); m.alloc(n); v.alloc(n);
    fill(g, n, 0.f); fill(m, n, 0.f); fill(v, n, 0.f);
  }
  void adam(float lr, int t, float wd, float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-8f) {
    float bc1 = 1.f - std::pow(b1, float(t)), bc2 = 1.f - std::pow(b2, float(t));
    float* W = w.data(); float* G = g.data(); float* M = m.data(); float* V = v.data();
    bk::parallel_for(n, [=] BK_HD (int64_t i) {
      float gr = G[i];
      M[i] = b1 * M[i] + (1.f - b1) * gr;
      V[i] = b2 * V[i] + (1.f - b2) * gr * gr;
      W[i] -= lr * ((M[i] / bc1) / (sqrtf(V[i] / bc2) + eps) + wd * W[i]);
    });
  }
};

// ---- model ------------------------------------------------------------------
struct Model {
  int H1 = 1024, H2 = 1024, NB = 2048;
  Param W1, b1, W2, b2, W3, b3;
  // activations / scratch, sized for maxB rows
  int maxB = 0;
  bk::Buffer<float> Z1, M1, A1, Z2, M2, A2, Z3, P, T, D3, DA2, DZ2, DA1, DZ1;
  bk::Buffer<int32_t> rowptr, col, ccolptr, ccol, crow;
  bk::Buffer<float> val, cval;

  static std::vector<float> randinit(int64_t n, float scale, uint64_t seed) {
    std::vector<float> h(n);
    uint64_t s = seed;
    for (auto& x : h) {
      s ^= s << 13; s ^= s >> 7; s ^= s << 17;
      float u1 = ((s >> 11) + 1) * (1.0f / 9007199254740993.0f);
      s ^= s << 13; s ^= s >> 7; s ^= s << 17;
      float u2 = (s >> 11) * (1.0f / 9007199254740992.0f);
      x = std::sqrt(-2.f * std::log(u1)) * std::cos(6.2831853f * u2) * scale;
    }
    return h;
  }
  void init(uint64_t seed = 7) {
    W1.init(int64_t(D_IN) * H1, randinit(int64_t(D_IN) * H1, std::sqrt(2.f / 64.f), seed));
    b1.init(H1, std::vector<float>(H1, 0.f));
    W2.init(int64_t(H1) * H2, randinit(int64_t(H1) * H2, std::sqrt(2.f / H1), seed + 1));
    b2.init(H2, std::vector<float>(H2, 0.f));
    W3.init(int64_t(H2) * NB, randinit(int64_t(H2) * NB, std::sqrt(1.f / H2), seed + 2));
    b3.init(NB, std::vector<float>(NB, -3.f));  // ~5% prior: most bits are off
  }
  void reserve(int B) {
    if (B <= maxB) return;
    maxB = B;
    for (auto* b : {&Z1, &M1, &A1, &DA1, &DZ1}) b->alloc(int64_t(B) * H1);
    for (auto* b : {&Z2, &M2, &A2, &DA2, &DZ2}) b->alloc(int64_t(B) * H2);
    for (auto* b : {&Z3, &P, &T, &D3}) b->alloc(int64_t(B) * NB);
  }
  std::vector<Param*> params() { return {&W1, &b1, &W2, &b2, &W3, &b3}; }

  // CSR of the batch (rows = spectra). For training also the CSC (by input column).
  void setBatch(const std::vector<std::vector<std::pair<int, float>>>& rows, bool withCsc) {
    std::vector<int32_t> rp(rows.size() + 1, 0), c;
    std::vector<float> v;
    for (size_t r = 0; r < rows.size(); ++r) {
      for (auto& e : rows[r]) { c.push_back(e.first); v.push_back(e.second); }
      rp[r + 1] = int32_t(c.size());
    }
    upload(rowptr, rp); upload(col, c); upload(val, v);
    if (!withCsc) return;
    std::vector<std::pair<int, int>> ent;  // (column, row) with value index
    std::vector<int> order(c.size());
    for (size_t i = 0; i < c.size(); ++i) order[i] = int(i);
    std::vector<int32_t> rowOf(c.size());
    for (size_t r = 0; r < rows.size(); ++r)
      for (int k = rp[r]; k < rp[r + 1]; ++k) rowOf[k] = int32_t(r);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return c[a] < c[b]; });
    std::vector<int32_t> cp, cc, cr;
    std::vector<float> cv;
    for (size_t i = 0; i < order.size(); ++i) {
      int k = order[i];
      if (cc.empty() || cc.back() != c[k]) { cc.push_back(c[k]); cp.push_back(int32_t(i)); }
      cr.push_back(rowOf[k]); cv.push_back(v[k]);
    }
    cp.push_back(int32_t(order.size()));
    nCols = int(cc.size());
    upload(ccolptr, cp); upload(ccol, cc); upload(crow, cr); upload(cval, cv);
  }
  int nCols = 0;

  // Forward. train=true applies dropout (rate p) using a hash of (seed, element).
  void forward(int B, bool train, float p, uint64_t seed) {
    reserve(B);
    const int h1 = H1, h2 = H2, nb = NB;
    const int32_t* RP = rowptr.data(); const int32_t* C = col.data(); const float* V = val.data();
    const float* w1 = W1.w.data(); const float* bb1 = b1.w.data();
    float* z1 = Z1.data(); float* m1 = M1.data(); float* a1 = A1.data();
    bk::parallel_for(int64_t(B) * h1, [=] BK_HD (int64_t idx) {
      int64_t r = idx / h1, j = idx % h1;
      float s = bb1[j];
      for (int k = RP[r]; k < RP[r + 1]; ++k) s += V[k] * w1[int64_t(C[k]) * h1 + j];
      z1[idx] = s;
    });
    actDropout(Z1, M1, A1, int64_t(B) * h1, train, p, seed * 2 + 1);
    bk::gemm(A1.data(), W2.w.data(), Z2.data(), B, h1, h2);
    addBias(Z2, b2, B, h2);
    actDropout(Z2, M2, A2, int64_t(B) * h2, train, p, seed * 2 + 2);
    bk::gemm(A2.data(), W3.w.data(), Z3.data(), B, h2, nb);
    addBias(Z3, b3, B, nb);
    float* z3 = Z3.data(); float* pp = P.data();
    bk::parallel_for(int64_t(B) * nb, [=] BK_HD (int64_t i) { pp[i] = 1.f / (1.f + expf(-z3[i])); });
  }

  void addBias(bk::Buffer<float>& Z, Param& b, int B, int n) {
    float* z = Z.data(); const float* bb = b.w.data();
    bk::parallel_for(int64_t(B) * n, [=] BK_HD (int64_t i) { z[i] += bb[i % n]; });
  }
  // a = relu(z) * mask/(1-p); M holds d a / d z for the backward pass.
  void actDropout(bk::Buffer<float>& Z, bk::Buffer<float>& M, bk::Buffer<float>& A,
                  int64_t n, bool train, float p, uint64_t seed) {
    const float* z = Z.data(); float* m = M.data(); float* a = A.data();
    const float keep = 1.f - p;
    bk::parallel_for(n, [=] BK_HD (int64_t i) {
      float d = z[i] > 0.f ? 1.f : 0.f;
      if (train && p > 0.f) {
        uint64_t x = uint64_t(i) * 0x9E3779B97F4A7C15ull + seed * 0xD1B54A32D192ED03ull;
        x ^= x >> 33; x *= 0xFF51AFD7ED558CCDull; x ^= x >> 33;
        float u = float(x >> 40) * (1.f / 16777216.f);
        d = (u < keep) ? d / keep : 0.f;
      }
      m[i] = d;
      a[i] = z[i] * d;
    });
  }

  // Backward for mean weighted BCE against T (dense 0/1, B x NB already uploaded).
  void backward(int B, float posWeight) {
    const int h1 = H1, h2 = H2, nb = NB;
    float* d3 = D3.data(); const float* pp = P.data(); const float* t = T.data();
    const float invB = 1.f / float(B), pw = posWeight;
    bk::parallel_for(int64_t(B) * nb, [=] BK_HD (int64_t i) {
      float tt = t[i];
      d3[i] = (pp[i] * (pw * tt + 1.f - tt) - pw * tt) * invB;
    });
    bk::gemm_tn(A2.data(), D3.data(), W3.g.data(), h2, nb, B);   // dW3 = A2^T D3
    colSum(D3, b3, B, nb);
    bk::gemm_nt(D3.data(), W3.w.data(), DA2.data(), B, h2, nb);  // dA2 = D3 W3^T
    mulInto(DA2, M2, DZ2, int64_t(B) * h2);
    bk::gemm_tn(A1.data(), DZ2.data(), W2.g.data(), h1, h2, B);
    colSum(DZ2, b2, B, h2);
    bk::gemm_nt(DZ2.data(), W2.w.data(), DA1.data(), B, h1, h2);
    mulInto(DA1, M1, DZ1, int64_t(B) * h1);
    colSum(DZ1, b1, B, h1);
    // dW1 = X^T DZ1, sparse: zero then fill only the touched rows (input columns).
    fill(W1.g, W1.n, 0.f);
    const int32_t* CP = ccolptr.data(); const int32_t* CC = ccol.data();
    const int32_t* CR = crow.data(); const float* CV = cval.data();
    const float* dz1 = DZ1.data(); float* gw1 = W1.g.data();
    bk::parallel_for(int64_t(nCols) * h1, [=] BK_HD (int64_t idx) {
      int64_t u = idx / h1, j = idx % h1;
      float s = 0.f;
      for (int k = CP[u]; k < CP[u + 1]; ++k) s += CV[k] * dz1[int64_t(CR[k]) * h1 + j];
      gw1[int64_t(CC[u]) * h1 + j] = s;
    });
  }
  void colSum(bk::Buffer<float>& D, Param& b, int B, int n) {
    const float* d = D.data(); float* g = b.g.data();
    bk::parallel_for(n, [=] BK_HD (int64_t j) {
      float s = 0.f;
      for (int r = 0; r < B; ++r) s += d[int64_t(r) * n + j];
      g[j] = s;
    });
  }
  void mulInto(bk::Buffer<float>& A, bk::Buffer<float>& Bm, bk::Buffer<float>& C, int64_t n) {
    const float* a = A.data(); const float* bb = Bm.data(); float* c = C.data();
    bk::parallel_for(n, [=] BK_HD (int64_t i) { c[i] = a[i] * bb[i]; });
  }

  void save(const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    int32_t hdr[5] = {0x56455246, D_IN, H1, H2, NB};
    f.write(reinterpret_cast<const char*>(hdr), sizeof(hdr));
    std::vector<float> h;
    for (Param* p : params()) {
      download(p->w, h, p->n);
      f.write(reinterpret_cast<const char*>(h.data()), h.size() * 4);
    }
  }
  bool load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    int32_t hdr[5];
    f.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
    if (hdr[0] != 0x56455246 || hdr[1] != D_IN) return false;
    H1 = hdr[2]; H2 = hdr[3]; NB = hdr[4];
    int64_t sizes[6] = {int64_t(D_IN) * H1, H1, int64_t(H1) * H2, H2, int64_t(H2) * NB, NB};
    auto ps = params();
    for (int i = 0; i < 6; ++i) {
      std::vector<float> h(sizes[i]);
      f.read(reinterpret_cast<char*>(h.data()), sizes[i] * 4);
      ps[i]->init(sizes[i], h);
    }
    return true;
  }
};

}  // namespace rev
