// cnn_layers.hpp : 1-D convolution layers on the mass axis for the CNN reverse
// model (casmi/cnn_STATUS.md). Hand-written forward + backward on the bk
// backend (same source on CUDA and CPU), checked against finite differences
// by cnn_gradcheck.cpp.
//
// Layout is channel-last: a signal batch is [N][L][C] row-major (N rows of a
// length-L axis with C channels). All ops keep their own small state; the
// caller owns activations.
//
//   Conv1d     : same-padded conv, kernel k (odd), stride 1, via im2col + GEMM
//                (W is [(k*Ci) x Co], row (j*Ci+ci)), + bias
//   relu       : in place; backward masks with the post-activation (y>0 <=> x>0)
//   maxpool2   : [N][L][C] -> [N][L/2][C]; backward recomputes the argmax
//   gpool      : [N][L][C] -> [N][2C] = (max over L, mean over L)
//   SparseStem : the first pyramid level. Each peak (or neutral loss, or pair
//                difference) is an item (row n, cell, fine, w): cell = 1 Da
//                coarse position, fine = position inside the cell on a
//                1/F Da grid. out[n][cell][c] = b[c] + sum_items w * Ws[fine][c]
//                = a conv at the fine resolution with kernel = stride = F
//                (patchify), evaluated only where items are.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "parallel.hpp"
#include "rev2_model.hpp"  // Head, upload/download, randinit (used unchanged)

// Scalar type of all CNN tensors: float for training/inference; the CPU
// gradient check builds with -DCNN_REAL=double so finite differences are
// not limited by float rounding (bk's GEMMs are float-only, so the double
// build uses the plain loops below).
#ifndef CNN_REAL
#define CNN_REAL float
#endif

namespace cnn {

using real = CNN_REAL;
using rev2::upload;
using rev2::download;

inline std::vector<real> randinit(int64_t n, float scale, uint64_t seed) {
  std::vector<float> h = rev2::randinit(n, scale, seed);
  return std::vector<real>(h.begin(), h.end());
}
inline void fill(bk::Buffer<real>& b, int64_t n, real v) {
  real* p = b.data();
  bk::parallel_for(n, [=] BK_HD (int64_t i) { p[i] = v; });
}
BK_HD inline real rexp(real x) {
  if constexpr (sizeof(real) == 4) return expf(x); else return exp(x);
}

// Same as rev2::Param (Adam state on the device) for the CNN scalar type.
struct Param {
  int64_t n = 0;
  bk::Buffer<real> w, g, m, v;
  void init(int64_t count, const std::vector<real>& host, bool train = true) {
    n = count;
    upload(w, host);
    if (!train) return;
    g.alloc(n); m.alloc(n); v.alloc(n);
    fill(g, n, 0); fill(m, n, 0); fill(v, n, 0);
  }
  void adam(float lr, int t, float wd, float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-8f) {
    real bc1 = real(1.f - std::pow(b1, float(t))), bc2 = real(1.f - std::pow(b2, float(t)));
    real* W = w.data(); real* G = g.data(); real* M = m.data(); real* V = v.data();
    bk::parallel_for(n, [=] BK_HD (int64_t i) {
      real gr = G[i];
      M[i] = b1 * M[i] + (1 - b1) * gr;
      V[i] = b2 * V[i] + (1 - b2) * gr * gr;
      W[i] -= lr * ((M[i] / bc1) / (sqrt(V[i] / bc2) + eps) + wd * W[i]);
    });
  }
};

// GEMMs (row-major, C = op(A) op(B) + beta C): bk for float, plain loops for double.
inline void gemm(const real* A, const real* B, real* C, int64_t M, int64_t K, int64_t N, real beta = 0) {
  if constexpr (sizeof(real) == 4) bk::gemm((const float*)A, (const float*)B, (float*)C, M, K, N, float(beta));
  else ::parallel_for(M, [=](int64_t m) {
    for (int64_t n = 0; n < N; ++n) { real s = 0; for (int64_t k = 0; k < K; ++k) s += A[m * K + k] * B[k * N + n];
      C[m * N + n] = s + (beta == 0 ? real(0) : beta * C[m * N + n]); } });
}
inline void gemm_nt(const real* A, const real* B, real* C, int64_t M, int64_t N, int64_t Kc, real beta = 0) {
  if constexpr (sizeof(real) == 4) bk::gemm_nt((const float*)A, (const float*)B, (float*)C, M, N, Kc, float(beta));
  else ::parallel_for(M, [=](int64_t m) {
    for (int64_t n = 0; n < N; ++n) { real s = 0; for (int64_t k = 0; k < Kc; ++k) s += A[m * Kc + k] * B[n * Kc + k];
      C[m * N + n] = s + (beta == 0 ? real(0) : beta * C[m * N + n]); } });
}
inline void gemm_tn(const real* A, const real* B, real* C, int64_t M, int64_t N, int64_t Kc, real beta = 0) {
  if constexpr (sizeof(real) == 4) bk::gemm_tn((const float*)A, (const float*)B, (float*)C, M, N, Kc, float(beta));
  else ::parallel_for(M, [=](int64_t m) {
    for (int64_t n = 0; n < N; ++n) { real s = 0; for (int64_t k = 0; k < Kc; ++k) s += A[k * M + m] * B[k * N + n];
      C[m * N + n] = s + (beta == 0 ? real(0) : beta * C[m * N + n]); } });
}

// ---- shared scratch -----------------------------------------------------
struct Scratch {
  bk::Buffer<real> col, dcol, ones, tmp;
  int64_t nOnes = 0;
  void need(bk::Buffer<real>& b, int64_t n) { if (b.size() < n || !b.data()) b.alloc(std::max<int64_t>(n, 1)); }
  const real* onesN(int64_t n) {
    if (nOnes < n) { need(ones, n); fill(ones, n, 1.f); nOnes = n; }
    return ones.data();
  }
};

// g[Co] (+)= sum over rows of D[rows][Co]   (bias gradient as a GEMM with ones)
inline void colsum(Scratch& s, const real* D, real* g, int64_t rows, int64_t Co, real beta) {
  gemm_tn(D, s.onesN(rows), g, Co, 1, rows, beta);
}

// ---- Conv1d ----------------------------------------------------------------
struct Conv1d {
  int Ci = 0, Co = 0, k = 5;
  Param W, b;
  void init(int ci, int co, int kk, uint64_t seed, bool train = true) {
    Ci = ci; Co = co; k = kk;
    int64_t n = int64_t(k) * Ci * Co;
    W.init(n, randinit(n, std::sqrt(2.f / real(k * Ci)), seed), train);
    b.init(Co, std::vector<real>(Co, 0.f), train);
  }
  void im2col(const real* X, real* col, int N, int L) const {
    const int ci = Ci, kk = k, pad = k / 2, KC = k * Ci;
    bk::parallel_for(int64_t(N) * L * KC, [=] BK_HD (int64_t idx) {
      int64_t r = idx / KC; int q = int(idx % KC);
      int j = q / ci, c = q % ci;
      int64_t n = r / L; int l = int(r % L);
      int src = l + j - pad;
      col[idx] = (src >= 0 && src < L) ? X[(n * L + src) * ci + c] : 0.f;
      (void)kk;
    });
  }
  // Y[N][L][Co] = conv(X[N][L][Ci]) + b
  void forward(Scratch& s, const real* X, real* Y, int N, int L) {
    const int64_t R = int64_t(N) * L, KC = int64_t(k) * Ci;
    s.need(s.col, R * KC);
    im2col(X, s.col.data(), N, L);
    gemm(s.col.data(), W.w.data(), Y, R, KC, Co);
    const real* bv = b.w.data(); const int co = Co;
    bk::parallel_for(R * Co, [=] BK_HD (int64_t i) { Y[i] += bv[i % co]; });
  }
  // dY = dLoss/d(pre-activation output). Accumulates W.g, b.g when acc=true
  // (else overwrites). Writes dX (overwrite) if dX != nullptr.
  void backward(Scratch& s, const real* X, const real* dY, real* dX, int N, int L, bool acc) {
    const int64_t R = int64_t(N) * L, KC = int64_t(k) * Ci;
    s.need(s.col, R * KC);
    im2col(X, s.col.data(), N, L);
    gemm_tn(s.col.data(), dY, W.g.data(), KC, Co, R, acc ? 1.f : 0.f);
    colsum(s, dY, b.g.data(), R, Co, acc ? 1.f : 0.f);
    if (!dX) return;
    s.need(s.dcol, R * KC);
    gemm_nt(dY, W.w.data(), s.dcol.data(), R, KC, Co);
    const real* dc = s.dcol.data();
    const int ci = Ci, kk = k, pad = k / 2;
    bk::parallel_for(R * Ci, [=] BK_HD (int64_t idx) {
      int64_t r = idx / ci; int c = int(idx % ci);
      int64_t n = r / L; int l = int(r % L);
      real acc2 = 0.f;
      for (int j = 0; j < kk; ++j) {
        int dst = l - j + pad;  // output position whose window holds input l at tap j
        if (dst >= 0 && dst < L) acc2 += dc[(n * L + dst) * (int64_t(kk) * ci) + j * ci + c];
      }
      dX[idx] = acc2;
    });
  }
};

inline void relu(real* Y, int64_t n) {
  bk::parallel_for(n, [=] BK_HD (int64_t i) { Y[i] = Y[i] > 0.f ? Y[i] : 0.f; });
}
// dX = dY * (Y > 0), in place on dY allowed
inline void reluBack(const real* Y, const real* dY, real* dX, int64_t n) {
  bk::parallel_for(n, [=] BK_HD (int64_t i) { dX[i] = Y[i] > 0.f ? dY[i] : 0.f; });
}

// ---- max pool by 2 ---------------------------------------------------------
inline void maxpool2(const real* X, real* Y, int N, int L, int C) {
  const int Lo = L / 2;
  bk::parallel_for(int64_t(N) * Lo * C, [=] BK_HD (int64_t idx) {
    int c = int(idx % C); int64_t r = idx / C; int64_t n = r / Lo; int l = int(r % Lo);
    real a = X[(n * L + 2 * l) * C + c], b2 = X[(n * L + 2 * l + 1) * C + c];
    Y[idx] = a >= b2 ? a : b2;
  });
}
inline void maxpool2Back(const real* X, const real* dY, real* dX, int N, int L, int C) {
  const int Lo = L / 2;
  bk::parallel_for(int64_t(N) * L * C, [=] BK_HD (int64_t idx) {
    int c = int(idx % C); int64_t r = idx / C; int64_t n = r / L; int l = int(r % L);
    int lo = l / 2;
    real a = X[(n * L + 2 * lo) * C + c], b2 = X[(n * L + 2 * lo + 1) * C + c];
    bool win = (l % 2 == 0) ? (a >= b2) : !(a >= b2);
    dX[idx] = win ? dY[(n * Lo + lo) * C + c] : 0.f;
  });
}

// ---- global pool: max and mean over the axis -------------------------------
// Y[n][0..C) = max, Y[n][C..2C) = mean ; Y row stride ys (to write into a wider matrix)
inline void gpool(const real* X, real* Y, int N, int L, int C, int64_t ys) {
  bk::parallel_for(int64_t(N) * C, [=] BK_HD (int64_t idx) {
    int64_t n = idx / C; int c = int(idx % C);
    real mx = -1e30f, sm = 0.f;
    for (int l = 0; l < L; ++l) { real v = X[(n * L + l) * C + c]; mx = v > mx ? v : mx; sm += v; }
    Y[n * ys + c] = mx; Y[n * ys + C + c] = sm / real(L);
  });
}
// dX[n][l][c] += dmax (at the first argmax) + dmean / L
inline void gpoolBackAdd(const real* X, const real* dY, real* dX, int N, int L, int C, int64_t ys) {
  bk::parallel_for(int64_t(N) * C, [=] BK_HD (int64_t idx) {
    int64_t n = idx / C; int c = int(idx % C);
    real mx = -1e30f; int am = 0;
    for (int l = 0; l < L; ++l) { real v = X[(n * L + l) * C + c]; if (v > mx) { mx = v; am = l; } }
    real dm = dY[n * ys + c], da = dY[n * ys + C + c] / real(L);
    for (int l = 0; l < L; ++l) dX[(n * L + l) * C + c] += da + (l == am ? dm : 0.f);
  });
}

// ---- sparse stem -------------------------------------------------------------
struct Item { int32_t row; int32_t cell; int32_t fine; real w; };  // row = n (batch row of the group)

struct SparseStem {
  int nFine = 0, C = 0;  // table rows = nStem * F (fine index already offset by stem id)
  Param W, b;
  // device copies of the batch's items: CSR by output row (n*L+cell), CSC by fine
  bk::Buffer<int32_t> rRow, rPtr, rFine, cFine, cPtr, cRow;
  bk::Buffer<real> rW, cW;
  int nR = 0, nCf = 0;
  void init(int fineRows, int c, uint64_t seed, bool train = true) {
    nFine = fineRows; C = c;
    W.init(int64_t(nFine) * C, randinit(int64_t(nFine) * C, 0.5f, seed), train);
    b.init(C, std::vector<real>(C, 0.f), train);
  }
  // Host side of a batch's items (built off the GPU thread, then uploaded).
  struct HostItems {
    std::vector<int32_t> rr, rp, rf, cf, cp, crow;
    std::vector<real> rw, cw;
    bool csc = false;
  };
  // items: any order; L = cells per row, N rows. Counting sorts (linear time).
  static void prepHost(const std::vector<Item>& items, int N, int L, int nFine, bool withCsc, HostItems& h) {
    const int64_t K = int64_t(N) * L;
    std::vector<int32_t> cnt(K + 1, 0);
    for (auto& it : items) ++cnt[int64_t(it.row) * L + it.cell + 1];
    h.rr.clear(); h.rp.clear();
    for (int64_t k = 0; k < K; ++k) {
      if (cnt[k + 1] > 0) { h.rr.push_back(int32_t(k)); h.rp.push_back(cnt[k]); }  // cnt[k] = start (prefix below)
      cnt[k + 1] += cnt[k];
    }
    // (at step k, cnt[k] already holds the start of key k; cnt is now the exclusive prefix)
    h.rp.push_back(int32_t(items.size()));
    h.rf.resize(items.size()); h.rw.resize(items.size());
    {
      std::vector<int32_t> pos(cnt.begin(), cnt.end() - 1);
      for (auto& it : items) {
        int32_t p = pos[int64_t(it.row) * L + it.cell]++;
        h.rf[p] = it.fine; h.rw[p] = it.w;
      }
    }
    h.csc = withCsc;
    if (!withCsc) return;
    std::vector<int32_t> fc(nFine + 1, 0);
    for (auto& it : items) ++fc[it.fine + 1];
    for (int f = 0; f < nFine; ++f) fc[f + 1] += fc[f];
    std::vector<int32_t> pos(fc.begin(), fc.end() - 1);
    h.crow.resize(items.size()); h.cw.resize(items.size());
    for (auto& it : items) { int32_t p = pos[it.fine]++; h.crow[p] = int32_t(int64_t(it.row) * L + it.cell); h.cw[p] = it.w; }
    h.cf.clear(); h.cp.clear();
    for (int f = 0; f < nFine; ++f) if (fc[f + 1] > fc[f]) { h.cf.push_back(f); h.cp.push_back(fc[f]); }
    h.cp.push_back(int32_t(items.size()));
  }
  void uploadItems(const HostItems& h) {
    nR = int(h.rr.size());
    upload(rRow, h.rr); upload(rPtr, h.rp); upload(rFine, h.rf); upload(rW, h.rw);
    nCf = 0;
    if (!h.csc) return;
    nCf = int(h.cf.size());
    upload(cFine, h.cf); upload(cPtr, h.cp); upload(cRow, h.crow); upload(cW, h.cw);
  }
  void setItems(const std::vector<Item>& items, int N, int L, bool withCsc) {
    HostItems h;
    prepHost(items, N, L, nFine, withCsc, h);
    uploadItems(h);
  }
  // Y[N][L][C] pre-activation
  void forward(real* Y, int N, int L) {
    const int c = C;
    const real* bv = b.w.data();
    bk::parallel_for(int64_t(N) * L * C, [=] BK_HD (int64_t i) { Y[i] = bv[i % c]; });
    if (!nR) return;
    const int32_t* RR = rRow.data(); const int32_t* RP = rPtr.data(); const int32_t* RF = rFine.data();
    const real* RW = rW.data(); const real* Wt = W.w.data();
    bk::parallel_for(int64_t(nR) * C, [=] BK_HD (int64_t idx) {
      int64_t u = idx / c; int ch = int(idx % c);
      real s = 0.f;
      for (int k = RP[u]; k < RP[u + 1]; ++k) s += RW[k] * Wt[int64_t(RF[k]) * c + ch];
      Y[int64_t(RR[u]) * c + ch] += s;
    });
  }
  // dY pre-activation [N][L][C]; accumulate or overwrite grads
  void backward(Scratch& s, const real* dY, int N, int L, bool acc) {
    colsum(s, dY, b.g.data(), int64_t(N) * L, C, acc ? 1.f : 0.f);
    if (!acc) fill(W.g, W.n, 0.f);
    if (!nCf) return;
    const int c = C;
    const int32_t* CF = cFine.data(); const int32_t* CP = cPtr.data(); const int32_t* CR = cRow.data();
    const real* CW = cW.data(); real* G = W.g.data();
    bk::parallel_for(int64_t(nCf) * C, [=] BK_HD (int64_t idx) {
      int64_t u = idx / c; int ch = int(idx % c);
      real sm = 0.f;
      for (int k = CP[u]; k < CP[u + 1]; ++k) sm += CW[k] * dY[int64_t(CR[k]) * c + ch];
      G[int64_t(CF[u]) * c + ch] += sm;
    });
  }
};

}  // namespace cnn
