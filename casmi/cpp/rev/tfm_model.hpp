// Spectrum -> fingerprint Transformer ("FPNet-like"), forward + hand-written
// backward on the own backend (backend.hpp): one source for CUDA (nvcc
// -DUSE_CUDA -DUSE_CUBLAS, cuBLAS strided-batched GEMMs, optional TF32) and
// the CPU (plain loops; -DTFM_DOUBLE gives a double build for exact
// finite-difference gradient checks).
//
// Architecture (after FPNet of ahmedberatozer/casmi26-v4b-models):
//   peak token  = Linear([SinEmb(mz), SinEmb(max(prec-mz,0)), it, it^2])   (it = sqrt(I/Imax))
//                 (FPNet's int_emb Linear(2,d) followed by pk is the same function class)
//   global token= Linear([SinEmb(prec), log1p(prec)/10, mode]) + E_adduct + E_instrument
//   L pre-LN blocks: x += drop(O(MHA(LN1 x)));  x += drop(W2 GELU(W1 LN2 x))
//   attention with key padding mask (batch padded to 1 + max peaks in the batch)
//   final LN; emb = [x_global, mean over peak tokens]; head Linear(2d,HH) GELU drop Linear(HH,NB)
//   loss: BCE with logits, mean over the batch, summed over bits.
// SinEmb(x)[i] = sin/cos(2 pi x / wav_i), wav_i = 10^((hi-lo) i/(n-1) + lo), n = demb/2.
//
// Layouts are row-major. Tokens of the batch: row r = b*N + n (n = 0 global,
// 1..np[b] peaks, rest padding). Attention tensors [B,H,N,hd].
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <type_traits>
#include <vector>

#include "backend.hpp"

namespace tfm {

#ifdef TFM_DOUBLE
using real = double;
#else
using real = float;
#endif
using Buf = bk::Buffer<real>;

// ---------------------------------------------------------------- launch helpers
#ifdef USE_CUDA
template <class F> __global__ void k_for(int64_t n, F f) {
  int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) f(i);
}
template <class F> inline void pfor(int64_t n, F f) {
  if (n <= 0) return;
  const int bs = 256;
  k_for<<<unsigned((n + bs - 1) / bs), bs>>>(n, f);
  BK_CHECK(cudaGetLastError());
}
// one warp per row; f(row, lane, nLanes) with warp reductions wsum/wmax
template <class F> __global__ void k_rows(int64_t rows, F f) {
  int64_t r = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
  int lane = threadIdx.x & 31;
  if (r < rows) f(r, lane, 32);
}
template <class F> inline void prows(int64_t rows, F f) {
  if (rows <= 0) return;
  const int bs = 256;
  int64_t th = rows * 32;
  k_rows<<<unsigned((th + bs - 1) / bs), bs>>>(rows, f);
  BK_CHECK(cudaGetLastError());
}
inline void sync() { BK_CHECK(cudaDeviceSynchronize()); }
#else
template <class F> inline void pfor(int64_t n, F f) { ::parallel_for(n, f); }
template <class F> inline void prows(int64_t rows, F f) {
  ::parallel_for(rows, [&](int64_t r) { f(r, 0, 1); });
}
inline void sync() {}
#endif

template <class T> BK_HD inline T wsum(T x) {
#ifdef __CUDA_ARCH__
  for (int o = 16; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
#endif
  return x;
}
template <class T> BK_HD inline T wmax(T x) {
#ifdef __CUDA_ARCH__
  for (int o = 16; o > 0; o >>= 1) { T y = __shfl_xor_sync(0xffffffffu, x, o); x = x > y ? x : y; }
#endif
  return x;
}

BK_HD inline float dropScale(int64_t i, uint64_t seed, float keep) {
  uint64_t x = uint64_t(i) * 0x9E3779B97F4A7C15ull + seed * 0xD1B54A32D192ED03ull;
  x ^= x >> 33; x *= 0xFF51AFD7ED558CCDull; x ^= x >> 33; x *= 0xC4CEB9FE1A85EC53ull; x ^= x >> 33;
  float u = float(x >> 40) * (1.f / 16777216.f);
  return u < keep ? 1.f / keep : 0.f;
}

// ---------------------------------------------------------------- host <-> device
template <class T> inline void upload(bk::Buffer<T>& b, const T* h, int64_t n, int64_t off = 0) {
  if (n <= 0) return;
#ifdef USE_CUDA
  BK_CHECK(cudaMemcpy(b.data() + off, h, size_t(n) * sizeof(T), cudaMemcpyHostToDevice));
#else
  std::memcpy(b.data() + off, h, size_t(n) * sizeof(T));
#endif
}
template <class T> inline void download(const bk::Buffer<T>& b, T* h, int64_t n, int64_t off = 0) {
  if (n <= 0) return;
#ifdef USE_CUDA
  BK_CHECK(cudaMemcpy(h, b.data() + off, size_t(n) * sizeof(T), cudaMemcpyDeviceToHost));
#else
  std::memcpy(h, b.data() + off, size_t(n) * sizeof(T));
#endif
}
template <class T> inline void ensure(bk::Buffer<T>& b, int64_t n) {
  if (b.size() < n || !b.data()) b.alloc(std::max<int64_t>(1, n));
}

// ---------------------------------------------------------------- GEMM
// Row-major C[M,N] = alpha op(A)[M,K] op(B)[K,N] + beta C, `batch` matrices with
// element strides sA/sB/sC. lda/ldb/ldc are the stored row lengths.
inline bool& tf32Flag() { static bool f = false; return f; }
#ifdef USE_CUDA
inline cublasHandle_t cublas() {
  static cublasHandle_t h = nullptr;
  static bool mode = false;
  if (!h) { cublasCreate(&h); mode = !tf32Flag(); }
  if (mode != tf32Flag()) {
    mode = tf32Flag();
    cublasSetMathMode(h, mode ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH);
  }
  return h;
}
#endif
inline void gemm(bool ta, bool tb, int M, int N, int K, real alpha, const real* A, int lda, int64_t sA,
                 const real* B, int ldb, int64_t sB, real beta, real* C, int ldc, int64_t sC, int batch = 1) {
#ifdef USE_CUDA
  cublasOperation_t oa = ta ? CUBLAS_OP_T : CUBLAS_OP_N, ob = tb ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasStatus_t st;
#ifdef TFM_DOUBLE
  st = cublasDgemmStridedBatched(cublas(), ob, oa, N, M, K, &alpha, B, ldb, sB, A, lda, sA, &beta, C, ldc, sC, batch);
#else
  st = cublasSgemmStridedBatched(cublas(), ob, oa, N, M, K, &alpha, B, ldb, sB, A, lda, sA, &beta, C, ldc, sC, batch);
#endif
  if (st != CUBLAS_STATUS_SUCCESS) { std::fprintf(stderr, "cublas error %d\n", int(st)); std::abort(); }
#else
  ::parallel_for(int64_t(batch) * M, [&](int64_t bm) {
    int64_t bi = bm / M; int m = int(bm % M);
    const real* a = A + bi * sA; const real* b = B + bi * sB; real* c = C + bi * sC + int64_t(m) * ldc;
    for (int n = 0; n < N; ++n) {
      real s = 0;
      for (int k = 0; k < K; ++k)
        s += (ta ? a[int64_t(k) * lda + m] : a[int64_t(m) * lda + k]) * (tb ? b[int64_t(n) * ldb + k] : b[int64_t(k) * ldb + n]);
      c[n] = alpha * s + (beta == real(0) ? real(0) : beta * c[n]);
    }
  });
#endif
}

// ---------------------------------------------------------------- config / params
struct Cfg {
  int d = 256, L = 4, H = 8, ff = 4, demb = 256, HH = 2048, NB = 4096, maxPeaks = 128;
  float lo = -2.f, hi = 3.3f;
  int nAd = 11, nIn = 6;  // 10 adducts + unknown, 5 instruments + unknown
  int fTok() const { return 2 * demb + 2; }
  int fGl() const { return demb + 2; }
};

struct PRef { int64_t off = 0, n = 0; int rows = 0, cols = 0; bool decay = false; const char* name = ""; };
struct BlockP { PRef ln1g, ln1b, wqkv, bqkv, wo, bo, ln2g, ln2b, w1, b1, w2, b2; };

struct HostBatch {
  int B = 0, N = 0;                  // N = 1 + max peaks
  std::vector<float> mz, it;         // B*(N-1), padded with 0
  std::vector<float> prec, mode;     // B
  std::vector<int32_t> np, ad, in;   // B
};

struct Model {
  Cfg c;
  PRef wtok, btok, wg, bg, ead, ein, lnfg, lnfb, wh1, bh1, wh2, bh2;
  std::vector<BlockP> blk;
  std::vector<PRef*> plist;
  int64_t nP = 0;
  Buf W, G, Mo, Vo;
  bk::Buffer<uint8_t> dec;
  bk::Buffer<double> iwav;  // 1/wav (demb/2)
  bool hasOpt = false;
  // vocabulary the output bits stand for (fp2; as in rev2 models)
  int rE = -1, rF = -1;
  std::vector<uint64_t> vocab;

  // ---- activations
  int capB = 0, capN = 0;
  int B = 0, N = 0;
  std::vector<Buf> X, xh1, rs1, Y1, QKV, P, A, xh2, rs2, Y2, Hp, Ha;
  Buf S3, Oh, tmp, Xmid, dT, dH, dY, dQKV, dP, dX, lnT;
  Buf Ftok, Fg, Ptok, Gt, dPtok, dGt;
  Buf xhf, rsf, Yf, Emb, Hh, Hha, Z, dZ, dHa, dEmb, dYf, ones, T;
  bk::Buffer<float> dmz, dit, dprec, dmode;
  bk::Buffer<int32_t> dnp, dad, din;

  PRef add(const char* name, int rows, int cols, bool decay) {
    PRef p; p.off = nP; p.rows = rows; p.cols = cols; p.n = int64_t(rows) * cols; p.decay = decay; p.name = name;
    nP += p.n;
    return p;
  }
  void layout() {
    nP = 0;
    const int d = c.d;
    wtok = add("wtok", c.fTok(), d, true); btok = add("btok", 1, d, false);
    wg = add("wg", c.fGl(), d, true); bg = add("bg", 1, d, false);
    ead = add("ead", c.nAd, d, false); ein = add("ein", c.nIn, d, false);
    blk.assign(c.L, BlockP());
    for (auto& b : blk) {
      b.ln1g = add("ln1g", 1, d, false); b.ln1b = add("ln1b", 1, d, false);
      b.wqkv = add("wqkv", d, 3 * d, true); b.bqkv = add("bqkv", 1, 3 * d, false);
      b.wo = add("wo", d, d, true); b.bo = add("bo", 1, d, false);
      b.ln2g = add("ln2g", 1, d, false); b.ln2b = add("ln2b", 1, d, false);
      b.w1 = add("w1", d, c.ff * d, true); b.b1 = add("b1", 1, c.ff * d, false);
      b.w2 = add("w2", c.ff * d, d, true); b.b2 = add("b2", 1, d, false);
    }
    lnfg = add("lnfg", 1, d, false); lnfb = add("lnfb", 1, d, false);
    wh1 = add("wh1", 2 * d, c.HH, true); bh1 = add("bh1", 1, c.HH, false);
    wh2 = add("wh2", c.HH, c.NB, true); bh2 = add("bh2", 1, c.NB, false);
    plist = {&wtok, &btok, &wg, &bg, &ead, &ein};
    for (auto& b : blk)
      for (PRef* p : {&b.ln1g, &b.ln1b, &b.wqkv, &b.bqkv, &b.wo, &b.bo, &b.ln2g, &b.ln2b, &b.w1, &b.b1, &b.w2, &b.b2})
        plist.push_back(p);
    for (PRef* p : {&lnfg, &lnfb, &wh1, &bh1, &wh2, &bh2}) plist.push_back(p);
    // sinusoid inverse wavelengths
    int n = c.demb / 2;
    std::vector<double> iw(n);
    for (int i = 0; i < n; ++i) {
      double t = n > 1 ? double(i) / double(n - 1) : 0.0;
      iw[i] = 1.0 / std::pow(10.0, double(c.hi - c.lo) * t + double(c.lo));
    }
    ensure(iwav, n);
    upload(iwav, iw.data(), n);
  }
  real* w(const PRef& p) { return W.data() + p.off; }
  real* g(const PRef& p) { return G.data() + p.off; }

  static void gauss(std::vector<real>& h, int64_t off, int64_t n, double sc, uint64_t& s) {
    for (int64_t i = 0; i < n; ++i) {
      s ^= s << 13; s ^= s >> 7; s ^= s << 17;
      double u1 = ((s >> 11) + 1) * (1.0 / 9007199254740993.0);
      s ^= s << 13; s ^= s >> 7; s ^= s << 17;
      double u2 = (s >> 11) * (1.0 / 9007199254740992.0);
      h[off + i] = real(std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2) * sc);
    }
  }
  // biasOut: per-bit initial output bias (logit of base rate), or empty -> -3
  void init(uint64_t seed, const std::vector<float>& biasOut, double std0 = 0.02) {
    layout();
    std::vector<real> h(nP, real(0));
    uint64_t s = seed * 0x9E3779B97F4A7C15ull + 7;
    const double sres = std0 / std::sqrt(2.0 * c.L);
    for (PRef* p : plist) {
      std::string nm = p->name;
      if (nm == "ln1g" || nm == "ln2g" || nm == "lnfg") for (int64_t i = 0; i < p->n; ++i) h[p->off + i] = 1;
      else if (nm == "wo" || nm == "w2") gauss(h, p->off, p->n, sres, s);
      else if (p->decay || nm == "ead" || nm == "ein") gauss(h, p->off, p->n, std0, s);
    }
    for (int i = 0; i < c.NB; ++i) h[bh2.off + i] = real(int(biasOut.size()) == c.NB ? biasOut[i] : -3.f);
    allocParams(true);
    upload(W, h.data(), nP);
  }
  void allocParams(bool opt) {
    ensure(W, nP); ensure(G, nP);
    std::vector<uint8_t> dm(nP, 0);
    for (PRef* p : plist) if (p->decay) std::fill(dm.begin() + p->off, dm.begin() + p->off + p->n, 1);
    ensure(dec, nP); upload(dec, dm.data(), nP);
    if (opt) {
      ensure(Mo, nP); ensure(Vo, nP);
      real* m = Mo.data(); real* v = Vo.data();
      pfor(nP, [=] BK_HD (int64_t i) { m[i] = 0; v[i] = 0; });
      hasOpt = true;
    }
  }

  void reserve(int Bc, int Nc) {
    if (Bc <= capB && Nc <= capN) return;
    capB = std::max(Bc, capB); capN = std::max(Nc, capN);
    const int64_t R = int64_t(capB) * capN, d = c.d, Hd = c.H;
    const int64_t att = int64_t(capB) * Hd * capN * capN;
    auto vec = [&](std::vector<Buf>& v, int64_t n) { v = std::vector<Buf>(c.L); for (auto& b : v) b.alloc(n); };
    vec(X, R * d); X = std::vector<Buf>(c.L + 1); for (auto& b : X) b.alloc(R * d);
    vec(xh1, R * d); vec(rs1, R); vec(Y1, R * d); vec(QKV, 3 * R * d); vec(P, att); vec(A, R * d);
    vec(xh2, R * d); vec(rs2, R); vec(Y2, R * d); vec(Hp, R * d * c.ff); vec(Ha, R * d * c.ff);
    for (Buf* b : {&S3, &dQKV}) b->alloc(3 * R * d);
    for (Buf* b : {&Oh, &tmp, &Xmid, &dT, &dY, &dX, &lnT, &xhf, &Yf, &dYf}) b->alloc(R * d);
    dH.alloc(R * d * c.ff); dP.alloc(att); rsf.alloc(R);
    const int64_t Rt = int64_t(capB) * (capN - 1);
    Ftok.alloc(Rt * c.fTok()); Ptok.alloc(Rt * d); dPtok.alloc(Rt * d);
    Fg.alloc(int64_t(capB) * c.fGl()); Gt.alloc(capB * d); dGt.alloc(capB * d);
    Emb.alloc(capB * 2 * d); dEmb.alloc(capB * 2 * d);
    Hh.alloc(int64_t(capB) * c.HH); Hha.alloc(int64_t(capB) * c.HH); dHa.alloc(int64_t(capB) * c.HH);
    Z.alloc(int64_t(capB) * c.NB); dZ.alloc(int64_t(capB) * c.NB); T.alloc(int64_t(capB) * c.NB);
    ones.alloc(R); { real* o = ones.data(); pfor(R, [=] BK_HD (int64_t i) { o[i] = 1; }); }
    dmz.alloc(Rt); dit.alloc(Rt); dprec.alloc(capB); dmode.alloc(capB);
    dnp.alloc(capB); dad.alloc(capB); din.alloc(capB);
  }

  void setBatch(const HostBatch& hb) {
    reserve(hb.B, hb.N);
    B = hb.B; N = hb.N;
    upload(dmz, hb.mz.data(), int64_t(B) * (N - 1)); upload(dit, hb.it.data(), int64_t(B) * (N - 1));
    upload(dprec, hb.prec.data(), B); upload(dmode, hb.mode.data(), B);
    upload(dnp, hb.np.data(), B); upload(dad, hb.ad.data(), B); upload(din, hb.in.data(), B);
  }

  // ------------------------------------------------------------ primitive ops
  void linear(const real* Xp, int64_t R, int in, const PRef& w_, const PRef& b_, real* Yp) {
    const int out = w_.cols;
    gemm(false, false, int(R), out, in, 1, Xp, in, 0, w(w_), out, 0, 0, Yp, out, 0);
    const real* bb = w(b_);
    pfor(R * out, [=] BK_HD (int64_t i) { Yp[i] += bb[i % out]; });
  }
  void colsum(const real* src, int64_t R, int n, real* dst) {
    gemm(false, false, 1, n, int(R), 1, ones.data(), int(R), 0, src, n, 0, 0, dst, n, 0);
  }
  // dW = X^T dY, db = colsum dY, dX (= or +=) dY W^T  (dXp may be null)
  void linearBack(const real* Xp, int64_t R, int in, const PRef& w_, const PRef& b_, const real* dYp, real* dXp,
                  real betaX = 0) {
    const int out = w_.cols;
    gemm(true, false, in, out, int(R), 1, Xp, in, 0, dYp, out, 0, 0, g(w_), out, 0);
    colsum(dYp, R, out, g(b_));
    if (dXp) gemm(false, true, int(R), in, out, 1, dYp, out, 0, w(w_), out, 0, betaX, dXp, in, 0);
  }
  void lnFwd(const real* Xp, const PRef& g_, const PRef& b_, real* Yp, real* XH, real* RS, int64_t R) {
    const int d = c.d; const real* gg = w(g_); const real* bb = w(b_);
    prows(R, [=] BK_HD (int64_t r, int lane, int nl) {
      const real* x = Xp + r * d;
      real s = 0;
      for (int j = lane; j < d; j += nl) s += x[j];
      real mu = wsum(s) / real(d);
      real v = 0;
      for (int j = lane; j < d; j += nl) { real t = x[j] - mu; v += t * t; }
      v = wsum(v) / real(d);
      real rs = real(1) / sqrt(v + real(1e-5));
      for (int j = lane; j < d; j += nl) {
        real hh = (x[j] - mu) * rs;
        XH[r * d + j] = hh; Yp[r * d + j] = hh * gg[j] + bb[j];
      }
      if (lane == 0) RS[r] = rs;
    });
  }
  // dX (+)= LN backward of dY; parameter grads into g_, b_
  void lnBack(const real* dYp, const real* XH, const real* RS, const PRef& g_, const PRef& b_, real* dXp, int64_t R,
              bool acc) {
    const int d = c.d; const real* gg = w(g_); real* T_ = lnT.data();
    prows(R, [=] BK_HD (int64_t r, int lane, int nl) {
      const real* dy = dYp + r * d; const real* xh = XH + r * d;
      real s1 = 0, s2 = 0;
      for (int j = lane; j < d; j += nl) { real t = dy[j] * gg[j]; s1 += t; s2 += t * xh[j]; }
      s1 = wsum(s1) / real(d); s2 = wsum(s2) / real(d);
      real rs = RS[r];
      for (int j = lane; j < d; j += nl) {
        real dx = rs * (dy[j] * gg[j] - s1 - xh[j] * s2);
        dXp[r * d + j] = acc ? dXp[r * d + j] + dx : dx;
        T_[r * d + j] = dy[j] * xh[j];
      }
    });
    colsum(lnT.data(), R, d, g(g_));
    colsum(dYp, R, d, g(b_));
  }
  static BK_HD inline real geluF(real x) { return real(0.5) * x * (real(1) + erf(x * real(0.70710678118654752))); }
  static BK_HD inline real geluD(real x) {
    return real(0.5) * (real(1) + erf(x * real(0.70710678118654752))) + x * real(0.3989422804014327) * exp(real(-0.5) * x * x);
  }
  void gelu(const real* Xp, real* Yp, int64_t n) { pfor(n, [=] BK_HD (int64_t i) { Yp[i] = geluF(Xp[i]); }); }
  void geluBack(const real* Xp, real* dYp, int64_t n) { pfor(n, [=] BK_HD (int64_t i) { dYp[i] *= geluD(Xp[i]); }); }
  void dropout(real* Yp, int64_t n, float p, uint64_t seed) {
    if (p <= 0.f) return;
    const float keep = 1.f - p;
    pfor(n, [=] BK_HD (int64_t i) { Yp[i] *= real(dropScale(i, seed, keep)); });
  }

  // ------------------------------------------------------------ forward
  // seed: per-step dropout seed. drop: residual/head dropout probability (train only)
  void forward(bool train, float drop, uint64_t seed) {
    const int d = c.d, Hn = c.H, hd = d / Hn, NB_ = N - 1, Ft = c.fTok(), Fgw = c.fGl(), hdem = c.demb / 2;
    const int64_t R = int64_t(B) * N, Rt = int64_t(B) * NB_;
    const float p = train ? drop : 0.f;
    // --- embeddings
    {
      const float* MZ = dmz.data(); const float* IT = dit.data(); const float* PR = dprec.data(); const float* MO = dmode.data();
      const int32_t* NP = dnp.data(); const double* IW = iwav.data(); real* F = Ftok.data(); real* FG = Fg.data();
      const int demb = c.demb;
      pfor(Rt * Ft, [=] BK_HD (int64_t idx) {
        int64_t t = idx / Ft; int cc = int(idx % Ft);
        int64_t b = t / NB_; int j = int(t % NB_);
        if (j >= NP[b]) { F[idx] = 0; return; }
        float m = MZ[t];
        if (cc < 2 * demb) {
          double x = cc < demb ? double(m) : double(fmaxf(PR[b] - m, 0.f));
          int k = cc % demb, i = k % hdem;
          double ph = x * IW[i]; ph -= floor(ph);
          real a = real(6.283185307179586 * ph);
          F[idx] = k < hdem ? sin(a) : cos(a);
        } else {
          float v = IT[t];
          F[idx] = cc == 2 * demb ? real(v) : real(v * v);
        }
      });
      pfor(int64_t(B) * Fgw, [=] BK_HD (int64_t idx) {
        int64_t b = idx / Fgw; int cc = int(idx % Fgw);
        if (cc < demb) {
          int i = cc % hdem;
          double ph = double(PR[b]) * IW[i]; ph -= floor(ph);
          real a = real(6.283185307179586 * ph);
          FG[idx] = cc < hdem ? sin(a) : cos(a);
        } else if (cc == demb) FG[idx] = real(log1p(double(PR[b])) / 10.0);
        else FG[idx] = real(MO[b]);
      });
      gemm(false, false, int(Rt), d, Ft, 1, Ftok.data(), Ft, 0, w(wtok), d, 0, 0, Ptok.data(), d, 0);
      gemm(false, false, B, d, Fgw, 1, Fg.data(), Fgw, 0, w(wg), d, 0, 0, Gt.data(), d, 0);
      real* X0 = X[0].data(); const real* PT = Ptok.data(); const real* GT = Gt.data();
      const real* BT = w(btok); const real* BG = w(bg); const real* EA = w(ead); const real* EI = w(ein);
      const int32_t* AD = dad.data(); const int32_t* IN = din.data(); const int Nn = N;
      pfor(R * d, [=] BK_HD (int64_t idx) {
        int64_t r = idx / d; int j = int(idx % d);
        int64_t b = r / Nn; int n = int(r % Nn);
        X0[idx] = n == 0 ? GT[b * d + j] + BG[j] + EA[int64_t(AD[b]) * d + j] + EI[int64_t(IN[b]) * d + j]
                         : PT[(b * NB_ + n - 1) * d + j] + BT[j];
      });
    }
    // --- blocks
    const real scale = real(1) / sqrt(real(hd));
    const int64_t BH = int64_t(B) * Hn, sQ = int64_t(N) * hd, sP = int64_t(N) * N;
    for (int l = 0; l < c.L; ++l) {
      const BlockP& bp = blk[l];
      lnFwd(X[l].data(), bp.ln1g, bp.ln1b, Y1[l].data(), xh1[l].data(), rs1[l].data(), R);
      linear(Y1[l].data(), R, d, bp.wqkv, bp.bqkv, S3.data());
      permQKV(S3.data(), QKV[l].data(), false);
      real* Q = QKV[l].data(); real* K = Q + R * d; real* V = K + R * d;
      gemm(false, true, N, N, hd, scale, Q, hd, sQ, K, hd, sQ, 0, P[l].data(), N, sP, int(BH));
      softmaxMask(P[l].data());
      gemm(false, false, N, hd, N, 1, P[l].data(), N, sP, V, hd, sQ, 0, Oh.data(), hd, sQ, int(BH));
      permO(Oh.data(), A[l].data(), false);
      linear(A[l].data(), R, d, bp.wo, bp.bo, tmp.data());
      dropout(tmp.data(), R * d, p, seed * 64 + l * 4 + 0);
      { real* xm = Xmid.data(); const real* x0 = X[l].data(); const real* t = tmp.data();
        pfor(R * d, [=] BK_HD (int64_t i) { xm[i] = x0[i] + t[i]; }); }
      lnFwd(Xmid.data(), bp.ln2g, bp.ln2b, Y2[l].data(), xh2[l].data(), rs2[l].data(), R);
      linear(Y2[l].data(), R, d, bp.w1, bp.b1, Hp[l].data());
      gelu(Hp[l].data(), Ha[l].data(), R * d * c.ff);
      linear(Ha[l].data(), R, d * c.ff, bp.w2, bp.b2, tmp.data());
      dropout(tmp.data(), R * d, p, seed * 64 + l * 4 + 1);
      { real* x1 = X[l + 1].data(); const real* xm = Xmid.data(); const real* t = tmp.data();
        pfor(R * d, [=] BK_HD (int64_t i) { x1[i] = xm[i] + t[i]; }); }
    }
    // --- head
    lnFwd(X[c.L].data(), lnfg, lnfb, Yf.data(), xhf.data(), rsf.data(), R);
    {
      const real* Y = Yf.data(); real* E = Emb.data(); const int32_t* NP = dnp.data(); const int Nn = N;
      pfor(int64_t(B) * d, [=] BK_HD (int64_t idx) {
        int64_t b = idx / d; int j = int(idx % d);
        E[b * 2 * d + j] = Y[(b * Nn) * d + j];
        real s = 0; int np = NP[b];
        for (int n = 1; n <= np; ++n) s += Y[(b * Nn + n) * d + j];
        E[b * 2 * d + d + j] = s / real(np > 0 ? np : 1);
      });
    }
    linear(Emb.data(), B, 2 * d, wh1, bh1, Hh.data());
    gelu(Hh.data(), Hha.data(), int64_t(B) * c.HH);
    dropout(Hha.data(), int64_t(B) * c.HH, p, seed * 64 + 63);
    linear(Hha.data(), B, c.HH, wh2, bh2, Z.data());
  }

  // qkv [R,3d] <-> QKV [3][B,H,N,hd]
  void permQKV(real* src, real* dst, bool back) {
    const int d = c.d, Hn = c.H, hd = d / Hn, Nn = N;
    const int64_t R = int64_t(B) * N;
    pfor(R * 3 * d, [=] BK_HD (int64_t idx) {
      int64_t r = idx / (3 * d); int cc = int(idx % (3 * d));
      int which = cc / d, k = cc % d, h = k / hd, e = k % hd;
      int64_t b = r / Nn; int n = int(r % Nn);
      int64_t o = which * R * d + ((b * Hn + h) * Nn + n) * hd + e;
      if (back) src[idx] = dst[o]; else dst[o] = src[idx];
    });
  }
  // O [B,H,N,hd] <-> A [R,d]
  void permO(real* oh, real* a, bool back) {
    const int d = c.d, Hn = c.H, hd = d / Hn, Nn = N;
    const int64_t R = int64_t(B) * N;
    pfor(R * d, [=] BK_HD (int64_t idx) {
      int64_t r = idx / d; int k = int(idx % d), h = k / hd, e = k % hd;
      int64_t b = r / Nn; int n = int(r % Nn);
      int64_t o = ((b * Hn + h) * Nn + n) * hd + e;
      if (back) oh[o] = a[idx]; else a[idx] = oh[o];
    });
  }
  void softmaxMask(real* Pp) {
    const int Hn = c.H, Nn = N; const int32_t* NP = dnp.data();
    prows(int64_t(B) * Hn * N, [=] BK_HD (int64_t r, int lane, int nl) {
      int64_t b = r / (int64_t(Hn) * Nn);
      int nv = 1 + NP[b];
      real* p = Pp + r * Nn;
      real mx = real(-1e30);
      for (int k = lane; k < nv; k += nl) mx = p[k] > mx ? p[k] : mx;
      mx = wmax(mx);
      real s = 0;
      for (int k = lane; k < nv; k += nl) { real e = exp(p[k] - mx); p[k] = e; s += e; }
      real inv = real(1) / wsum(s);
      for (int k = lane; k < Nn; k += nl) p[k] = k < nv ? p[k] * inv : real(0);
    });
  }

  // ------------------------------------------------------------ loss
  // dZ = (sigmoid(Z) - t') / B  with label smoothing ls
  void lossGrad(float ls) {
    real* dz = dZ.data(); const real* z = Z.data(); const real* t = T.data();
    const real invB = real(1) / real(B), l = real(ls);
    pfor(int64_t(B) * c.NB, [=] BK_HD (int64_t i) {
      real tt = t[i] * (real(1) - l) + real(0.5) * l;
      real pp = real(1) / (real(1) + exp(-z[i]));
      dz[i] = (pp - tt) * invB;
    });
  }

  // ------------------------------------------------------------ backward
  void backward(float drop, uint64_t seed) {
    const int d = c.d, Hn = c.H, hd = d / Hn, NB_ = N - 1, Ft = c.fTok(), Fgw = c.fGl();
    const int64_t R = int64_t(B) * N, Rt = int64_t(B) * NB_;
    const float p = drop;
    // head
    linearBack(Hha.data(), B, c.HH, wh2, bh2, dZ.data(), dHa.data());
    dropout(dHa.data(), int64_t(B) * c.HH, p, seed * 64 + 63);
    geluBack(Hh.data(), dHa.data(), int64_t(B) * c.HH);
    linearBack(Emb.data(), B, 2 * d, wh1, bh1, dHa.data(), dEmb.data());
    {
      const real* dE = dEmb.data(); real* dYp = dYf.data(); const int32_t* NP = dnp.data(); const int Nn = N;
      pfor(R * d, [=] BK_HD (int64_t idx) {
        int64_t r = idx / d; int j = int(idx % d);
        int64_t b = r / Nn; int n = int(r % Nn); int np = NP[b];
        dYp[idx] = n == 0 ? dE[b * 2 * d + j] : (n <= np ? dE[b * 2 * d + d + j] / real(np) : real(0));
      });
    }
    lnBack(dYf.data(), xhf.data(), rsf.data(), lnfg, lnfb, dX.data(), R, false);
    // blocks
    const real scale = real(1) / sqrt(real(hd));
    const int64_t BH = int64_t(B) * Hn, sQ = int64_t(N) * hd, sP = int64_t(N) * N;
    for (int l = c.L - 1; l >= 0; --l) {
      const BlockP& bp = blk[l];
      // FFN branch
      { real* t = dT.data(); const real* x = dX.data(); pfor(R * d, [=] BK_HD (int64_t i) { t[i] = x[i]; }); }
      dropout(dT.data(), R * d, p, seed * 64 + l * 4 + 1);
      linearBack(Ha[l].data(), R, d * c.ff, bp.w2, bp.b2, dT.data(), dH.data());
      geluBack(Hp[l].data(), dH.data(), R * d * c.ff);
      linearBack(Y2[l].data(), R, d, bp.w1, bp.b1, dH.data(), dY.data());
      lnBack(dY.data(), xh2[l].data(), rs2[l].data(), bp.ln2g, bp.ln2b, dX.data(), R, true);
      // attention branch
      { real* t = dT.data(); const real* x = dX.data(); pfor(R * d, [=] BK_HD (int64_t i) { t[i] = x[i]; }); }
      dropout(dT.data(), R * d, p, seed * 64 + l * 4 + 0);
      linearBack(A[l].data(), R, d, bp.wo, bp.bo, dT.data(), dY.data());
      permO(Oh.data(), dY.data(), true);  // Oh := dO
      real* Q = QKV[l].data(); real* K = Q + R * d; real* V = K + R * d;
      real* dQ = dQKV.data(); real* dK = dQ + R * d; real* dV = dK + R * d;
      gemm(true, false, N, hd, N, 1, P[l].data(), N, sP, Oh.data(), hd, sQ, 0, dV, hd, sQ, int(BH));
      gemm(false, true, N, N, hd, 1, Oh.data(), hd, sQ, V, hd, sQ, 0, dP.data(), N, sP, int(BH));
      {
        real* dp = dP.data(); const real* pp = P[l].data(); const int Nn = N;
        prows(BH * N, [=] BK_HD (int64_t r, int lane, int nl) {
          real* g_ = dp + r * Nn; const real* q = pp + r * Nn;
          real s = 0;
          for (int k = lane; k < Nn; k += nl) s += g_[k] * q[k];
          s = wsum(s);
          for (int k = lane; k < Nn; k += nl) g_[k] = q[k] * (g_[k] - s);
        });
      }
      gemm(false, false, N, hd, N, scale, dP.data(), N, sP, K, hd, sQ, 0, dQ, hd, sQ, int(BH));
      gemm(true, false, N, hd, N, scale, dP.data(), N, sP, Q, hd, sQ, 0, dK, hd, sQ, int(BH));
      permQKV(S3.data(), dQKV.data(), true);  // S3 := dqkv [R,3d]
      linearBack(Y1[l].data(), R, d, bp.wqkv, bp.bqkv, S3.data(), dY.data());
      lnBack(dY.data(), xh1[l].data(), rs1[l].data(), bp.ln1g, bp.ln1b, dX.data(), R, true);
    }
    // embeddings
    {
      const real* dx = dX.data(); real* dpt = dPtok.data(); real* dg = dGt.data(); const int Nn = N;
      pfor(Rt * d, [=] BK_HD (int64_t idx) {
        int64_t t = idx / d; int j = int(idx % d);
        int64_t b = t / NB_; int n = int(t % NB_);
        dpt[idx] = dx[(b * Nn + 1 + n) * d + j];
      });
      pfor(int64_t(B) * d, [=] BK_HD (int64_t idx) {
        int64_t b = idx / d; int j = int(idx % d);
        dg[idx] = dx[(b * Nn) * d + j];
      });
      gemm(true, false, Ft, d, int(Rt), 1, Ftok.data(), Ft, 0, dPtok.data(), d, 0, 0, g(wtok), d, 0);
      colsum(dPtok.data(), Rt, d, g(btok));
      gemm(true, false, Fgw, d, B, 1, Fg.data(), Fgw, 0, dGt.data(), d, 0, 0, g(wg), d, 0);
      colsum(dGt.data(), B, d, g(bg));
      const int32_t* AD = dad.data(); const int32_t* IN = din.data(); const int Bn = B;
      real* gea = g(ead); real* gei = g(ein); const int nAd = c.nAd;
      pfor(int64_t(c.nAd + c.nIn) * d, [=] BK_HD (int64_t idx) {
        int a = int(idx / d), j = int(idx % d);
        bool isAd = a < nAd; int k = isAd ? a : a - nAd;
        const int32_t* lab = isAd ? AD : IN;
        real s = 0;
        for (int b = 0; b < Bn; ++b) if (lab[b] == k) s += dg[int64_t(b) * d + j];
        (isAd ? gea : gei)[int64_t(k) * d + j] = s;
      });
    }
  }

  // ------------------------------------------------------------ optimizer
  double gradNorm2() {
    const int NPART = 4096;
    static Buf part;
    ensure(part, NPART);
    const real* G_ = G.data(); real* pt = part.data(); const int64_t n = nP;
    pfor(NPART, [=] BK_HD (int64_t pi) {
      real s = 0;
      for (int64_t i = pi; i < n; i += NPART) s += G_[i] * G_[i];
      pt[pi] = s;
    });
    std::vector<real> h(NPART);
    download(part, h.data(), NPART);
    double s = 0;
    for (real x : h) s += double(x);
    return s;
  }
  int64_t adamT = 0;
  void adamw(float lr, float wd, float gscale, float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-8f) {
    ++adamT;
    const real bc1 = real(1 - std::pow(double(b1), double(adamT))), bc2 = real(1 - std::pow(double(b2), double(adamT)));
    real* W_ = W.data(); const real* G_ = G.data(); real* M_ = Mo.data(); real* V_ = Vo.data(); const uint8_t* D = dec.data();
    const real lr_ = lr, wd_ = wd, gs = gscale, B1 = b1, B2 = b2, E = eps;
    pfor(nP, [=] BK_HD (int64_t i) {
      real gr = G_[i] * gs;
      real m = B1 * M_[i] + (1 - B1) * gr, v = B2 * V_[i] + (1 - B2) * gr * gr;
      M_[i] = m; V_[i] = v;
      W_[i] -= lr_ * ((m / bc1) / (sqrt(v / bc2) + E) + (D[i] ? wd_ * W_[i] : real(0)));
    });
  }

  // ------------------------------------------------------------ file format TFM1
  // int32 magic 'TFM1', version 1; Cfg ints d L H ff demb HH NB maxPeaks nAd nIn; float lo hi;
  // int64 nP; float W[nP]; int32 rE rF nVocab; uint64 vocab[]; int32 hasOpt; [int64 adamT; float M[nP], V[nP]]
  void save(const std::string& path, bool withOpt) {
    std::string tmpP = path + ".tmp";
    {
      std::ofstream f(tmpP, std::ios::binary);
      auto wi = [&](int32_t x) { f.write(reinterpret_cast<const char*>(&x), 4); };
      wi(0x314D4654); wi(1);
      for (int x : {c.d, c.L, c.H, c.ff, c.demb, c.HH, c.NB, c.maxPeaks, c.nAd, c.nIn}) wi(x);
      f.write(reinterpret_cast<const char*>(&c.lo), 4); f.write(reinterpret_cast<const char*>(&c.hi), 4);
      f.write(reinterpret_cast<const char*>(&nP), 8);
      auto wbuf = [&](const Buf& b) {
        std::vector<real> h(nP); download(b, h.data(), nP);
        std::vector<float> hf(h.begin(), h.end());
        f.write(reinterpret_cast<const char*>(hf.data()), 4 * nP);
      };
      wbuf(W);
      wi(rE); wi(rF); wi(int32_t(vocab.size()));
      if (!vocab.empty()) f.write(reinterpret_cast<const char*>(vocab.data()), 8 * vocab.size());
      bool wo = withOpt && hasOpt;
      wi(wo ? 1 : 0);
      if (wo) { f.write(reinterpret_cast<const char*>(&adamT), 8); wbuf(Mo); wbuf(Vo); }
    }
    std::remove(path.c_str());
    std::rename(tmpP.c_str(), path.c_str());
  }
  bool load(const std::string& path, bool withOpt) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    auto ri = [&]() { int32_t x = 0; f.read(reinterpret_cast<char*>(&x), 4); return x; };
    if (ri() != 0x314D4654) return false;
    ri();
    for (int* x : {&c.d, &c.L, &c.H, &c.ff, &c.demb, &c.HH, &c.NB, &c.maxPeaks, &c.nAd, &c.nIn}) *x = ri();
    f.read(reinterpret_cast<char*>(&c.lo), 4); f.read(reinterpret_cast<char*>(&c.hi), 4);
    int64_t n = 0; f.read(reinterpret_cast<char*>(&n), 8);
    layout();
    if (n != nP) { std::fprintf(stderr, "param count mismatch %lld vs %lld\n", (long long)n, (long long)nP); return false; }
    auto rbuf = [&](Buf& b) {
      std::vector<float> hf(nP); f.read(reinterpret_cast<char*>(hf.data()), 4 * nP);
      std::vector<real> h(hf.begin(), hf.end()); upload(b, h.data(), nP);
    };
    allocParams(withOpt);
    rbuf(W);
    rE = ri(); rF = ri(); int nv = ri();
    vocab.resize(nv);
    if (nv) f.read(reinterpret_cast<char*>(vocab.data()), 8 * size_t(nv));
    int ho = ri();
    if (withOpt && ho) { f.read(reinterpret_cast<char*>(&adamT), 8); rbuf(Mo); rbuf(Vo); }
    else adamT = 0;
    return bool(f);
  }
};

}  // namespace tfm
