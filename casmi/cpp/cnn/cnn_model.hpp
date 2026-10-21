// cnn_model.hpp : CNN reverse model (spectrum -> fingerprint-bit probabilities)
// with translation symmetry along the mass axis. Design and reasoning:
// casmi/cnn_STATUS.md. Output bits / vocabulary / calibration / file tail are
// the same as REV2 (rev2_model.hpp), so the REV2 tools' scoring applies as is.
//
// Three 1-D signals per spectrum, each on a 1 Da coarse axis with a 1/F Da
// fine position inside every cell (the sparse stem = first pyramid level):
//   frag : fragment m/z                          0 .. Lf Da
//   nl   : neutral loss  precursor - m/z         0 .. Ln Da
//   pair : |m/z_i - m/z_j| between the top-K peaks  0 .. Lp Da  (a fragment
//          losing a neutral = the same algebra as the precursor losing it)
// frag has its own stack; nl and pair share ONE conv stack (same weights,
// rows 0..B-1 = nl, B..2B-1 = pair; separate stems), because a mass difference
// means the same chemistry whether the heavier partner is the precursor or a
// fragment. Stack = [conv k, ReLU, maxpool 2] x nl (coarse-to-fine pyramid).
// Readout: global max+mean pool of each signal's top map (translation
// invariant) and, with flat=1, the flattened top map (keeps absolute
// position at 2^nl Da resolution), + adduct/instrument/precursor side
// features -> dense head (ReLU + dropout) -> NB logits.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "cnn_layers.hpp"

namespace cnn {

using rev2::Head;
constexpr int N_SIDE = rev2::N_ADDUCT + rev2::N_INSTR + 1;

struct Cfg {
  int F = 100;          // fine positions per Da (0.01 Da)
  float off = 0.3f;     // cell = floor(v + off): mass defects -0.3 .. +0.7 stay in their nominal cell
  int Lf = 1024, Ln = 512, Lp = 512;
  int C0 = 32, k = 5;
  std::vector<int> chans{64, 64, 128, 128};
  int flat = 1;
  std::vector<int> H{3072};
  int maxPeaks = 300, pairK = 60;
  int nl() const { return int(chans.size()); }
  int top(int L) const { return L >> nl(); }
  int Ctop() const { return chans.back(); }
  int64_t dHead() const {
    int64_t d = 3 * 2 * int64_t(Ctop()) + N_SIDE;
    if (flat) d += int64_t(top(Lf) + top(Ln) + top(Lp)) * Ctop();
    return d;
  }
};

struct Sample {
  std::vector<Item> frag, nl, pair;  // row field unused here (0)
  float side[N_SIDE];
};

inline void putItem(std::vector<Item>& v, float x, float w, const Cfg& c, int L, int fineOff) {
  float y = x + c.off;
  if (y < 0.f) return;
  int cell = int(std::floor(y));
  if (cell >= L) return;
  int fine = int((y - float(cell)) * float(c.F));
  fine = std::min(c.F - 1, std::max(0, fine));
  v.push_back({0, cell, fineOff + fine, w});
}

inline void featurize(const Cfg& c, const Head& h, const float* mz, const float* it, Sample& s) {
  s.frag.clear(); s.nl.clear(); s.pair.clear();
  int n = h.n;
  std::vector<int> idx(n);
  for (int i = 0; i < n; ++i) idx[i] = i;
  std::sort(idx.begin(), idx.end(), [&](int a, int b) { return it[a] > it[b]; });
  if (n > c.maxPeaks) idx.resize(c.maxPeaks);
  float mx = 0.f;
  for (int i : idx) mx = std::max(mx, it[i]);
  if (mx <= 0.f) mx = 1.f;
  std::vector<float> w(idx.size());
  for (size_t t = 0; t < idx.size(); ++t) {
    int i = idx[t];
    w[t] = std::sqrt(std::max(0.f, it[i]) / mx);
    putItem(s.frag, mz[i], w[t], c, c.Lf, 0);
    float nlv = h.precursor - mz[i];
    if (nlv > -c.off) putItem(s.nl, std::max(0.f, nlv), w[t], c, c.Ln, 0);
  }
  size_t K = std::min<size_t>(idx.size(), size_t(c.pairK));
  for (size_t a = 0; a < K; ++a)
    for (size_t b = a + 1; b < K; ++b) {
      float d = std::fabs(mz[idx[a]] - mz[idx[b]]);
      if (d < 0.5f) continue;
      putItem(s.pair, d, w[a] * w[b], c, c.Lp, c.F);
    }
  for (float& x : s.side) x = 0.f;
  if (h.tag >= 0) {
    bool unkInstr = h.tag % 16 == 15;
    int a = unkInstr ? h.tag / 16 + 1 : h.tag / 16;
    int sI = unkInstr ? -1 : h.tag % 16;
    if (a >= 0 && a < rev2::N_ADDUCT) s.side[a] = 1.f;
    if (sI >= 0 && sI < rev2::N_INSTR) s.side[rev2::N_ADDUCT + sI] = 1.f;
  }
  s.side[N_SIDE - 1] = h.precursor / 1000.f;
}

// ---- conv stack ---------------------------------------------------------------
struct Stack {
  SparseStem stem;
  std::vector<std::unique_ptr<Conv1d>> conv;
  int L0 = 0, maxN = 0;
  std::vector<std::unique_ptr<bk::Buffer<real>>> X, Y, dX;  // X[0..nl], Y[0..nl-1], dX[0..nl]
  bk::Buffer<real> gY;
  std::vector<int> C;  // channels per level: C[0]=C0, C[i+1]=chans[i]
  int nl() const { return int(conv.size()); }
  int Lat(int i) const { return L0 >> i; }
  void init(const Cfg& c, int L, int fineRows, uint64_t seed, bool train) {
    L0 = L;
    C.assign(1, c.C0);
    for (int ch : c.chans) C.push_back(ch);
    stem.init(fineRows, c.C0, seed, train);
    conv.clear();
    for (int i = 0; i < c.nl(); ++i) {
      conv.emplace_back(new Conv1d());
      conv.back()->init(C[i], C[i + 1], c.k, seed + 101 + i, train);
    }
    maxN = 0;
  }
  void reserve(int N) {
    if (N <= maxN) return;
    maxN = N;
    X.clear(); Y.clear(); dX.clear();
    for (int i = 0; i <= nl(); ++i) {
      X.emplace_back(new bk::Buffer<real>(int64_t(N) * Lat(i) * C[i]));
      dX.emplace_back(new bk::Buffer<real>(int64_t(N) * Lat(i) * C[i]));
    }
    int64_t mY = 1;
    for (int i = 0; i < nl(); ++i) {
      Y.emplace_back(new bk::Buffer<real>(int64_t(N) * Lat(i) * C[i + 1]));
      mY = std::max(mY, int64_t(N) * Lat(i) * C[i + 1]);
    }
    gY.alloc(mY);
  }
  const real* top() const { return X[nl()]->data(); }
  real* dTop() { return dX[nl()]->data(); }
  int Ltop() const { return Lat(nl()); }
  void forward(Scratch& s, int N) {
    reserve(N);
    stem.forward(X[0]->data(), N, L0);
    relu(X[0]->data(), int64_t(N) * L0 * C[0]);
    for (int i = 0; i < nl(); ++i) {
      int L = Lat(i);
      conv[i]->forward(s, X[i]->data(), Y[i]->data(), N, L);
      relu(Y[i]->data(), int64_t(N) * L * C[i + 1]);
      maxpool2(Y[i]->data(), X[i + 1]->data(), N, L, C[i + 1]);
    }
  }
  // dTop() must hold dLoss/dX[nl]. Overwrites parameter grads.
  void backward(Scratch& s, int N) {
    for (int i = nl() - 1; i >= 0; --i) {
      int L = Lat(i);
      int64_t nY = int64_t(N) * L * C[i + 1];
      maxpool2Back(Y[i]->data(), dX[i + 1]->data(), gY.data(), N, L, C[i + 1]);
      reluBack(Y[i]->data(), gY.data(), gY.data(), nY);
      conv[i]->backward(s, X[i]->data(), gY.data(), dX[i]->data(), N, L, false);
    }
    reluBack(X[0]->data(), dX[0]->data(), dX[0]->data(), int64_t(N) * L0 * C[0]);
    stem.backward(s, dX[0]->data(), N, L0, false);
  }
  void params(std::vector<Param*>& p) {
    p.push_back(&stem.W); p.push_back(&stem.b);
    for (auto& cv : conv) { p.push_back(&cv->W); p.push_back(&cv->b); }
  }
};

// ---- dense layer ----------------------------------------------------------------
struct Dense {
  int ni = 0, no = 0;
  Param W, b;
  void init(int i, int o, real sc, real bias, uint64_t seed, bool train) {
    ni = i; no = o;
    W.init(int64_t(ni) * no, randinit(int64_t(ni) * no, sc, seed), train);
    b.init(no, std::vector<real>(no, bias), train);
  }
  void forward(const real* X, real* Y, int B) {
    gemm(X, W.w.data(), Y, B, ni, no);
    const real* bv = b.w.data(); const int n = no;
    bk::parallel_for(int64_t(B) * no, [=] BK_HD (int64_t i) { Y[i] += bv[i % n]; });
  }
  void backward(Scratch& s, const real* X, const real* dY, real* dX, int B) {
    gemm_tn(X, dY, W.g.data(), ni, no, B);
    colsum(s, dY, b.g.data(), B, no, 0.f);
    if (dX) gemm_nt(dY, W.w.data(), dX, B, ni, no);
  }
};

struct Model {
  Cfg cfg;
  int NB = 4096;
  int rE = -1, rF = -1;
  std::vector<uint64_t> vocab;
  std::vector<float> calA, calC, quality;

  Stack sf, sl;  // fragment stack; shared loss stack (nl rows 0..B-1, pair rows B..2B-1)
  std::vector<std::unique_ptr<Dense>> head;
  Scratch scr;
  int maxB = 0;
  bk::Buffer<real> Hin, dHin, side, Zo, P, T, Dout;
  std::vector<std::unique_ptr<bk::Buffer<real>>> HA, HM, HD;  // hidden activation, dropout mask, grad
  real* extraGrad = nullptr;  // optional dLoss/dlogit added in backward (unused by default)

  int64_t D() const { return cfg.dHead(); }
  void build(uint64_t seed, bool train) {
    sf.init(cfg, cfg.Lf, cfg.F, seed + 1000, train);
    sl.init(cfg, cfg.Ln, 2 * cfg.F, seed + 2000, train);
    if (cfg.Lp != cfg.Ln) { std::fprintf(stderr, "cnn: Lp must equal Ln (shared stack)\n"); std::exit(1); }
    head.clear();
    int64_t in = D();
    for (size_t l = 0; l <= cfg.H.size(); ++l) {
      int out = l < cfg.H.size() ? cfg.H[l] : NB;
      bool last = l == cfg.H.size();
      head.emplace_back(new Dense());
      head.back()->init(int(in), out, last ? std::sqrt(1.f / real(in)) : std::sqrt(2.f / real(in)),
                        last ? -3.f : 0.f, seed + 3000 + l, train);
      in = out;
    }
    maxB = 0;
  }
  std::vector<Param*> params() {
    std::vector<Param*> p;
    sf.params(p); sl.params(p);
    for (auto& d : head) { p.push_back(&d->W); p.push_back(&d->b); }
    return p;
  }
  int64_t nParams() { int64_t n = 0; for (Param* p : params()) n += p->n; return n; }
  void reserve(int B) {
    if (B <= maxB) return;
    maxB = B;
    Hin.alloc(int64_t(B) * D()); dHin.alloc(int64_t(B) * D()); side.alloc(int64_t(B) * N_SIDE);
    HA.clear(); HM.clear(); HD.clear();
    for (int h : cfg.H) {
      HA.emplace_back(new bk::Buffer<real>(int64_t(B) * h));
      HM.emplace_back(new bk::Buffer<real>(int64_t(B) * h));
      HD.emplace_back(new bk::Buffer<real>(int64_t(B) * h));
    }
    for (auto* v : {&Zo, &P, &T, &Dout}) v->alloc(int64_t(B) * NB);
  }

  // Host part of a batch (thread-safe, no device access): run it in the prefetch thread.
  struct HostBatch { int B = 0; SparseStem::HostItems f, l; std::vector<real> sd; };
  void prepBatch(const std::vector<Sample>& ss, bool train, HostBatch& hb) const {
    const int B = int(ss.size());
    hb.B = B;
    std::vector<Item> f, l;
    hb.sd.assign(size_t(B) * N_SIDE, 0);
    for (int b = 0; b < B; ++b) {
      for (Item it : ss[b].frag) { it.row = b; f.push_back(it); }
      for (Item it : ss[b].nl) { it.row = b; l.push_back(it); }
      for (Item it : ss[b].pair) { it.row = B + b; l.push_back(it); }
      std::copy(ss[b].side, ss[b].side + N_SIDE, hb.sd.begin() + size_t(b) * N_SIDE);
    }
    SparseStem::prepHost(f, B, cfg.Lf, sf.stem.nFine, train, hb.f);
    SparseStem::prepHost(l, 2 * B, cfg.Ln, sl.stem.nFine, train, hb.l);
  }
  void uploadBatch(const HostBatch& hb) {
    reserve(hb.B);
    sf.stem.uploadItems(hb.f);
    sl.stem.uploadItems(hb.l);
    upload(side, hb.sd);
  }
  void setBatch(const std::vector<Sample>& ss, bool train) {
    HostBatch hb;
    prepBatch(ss, train, hb);
    uploadBatch(hb);
  }

  // segment offsets in the head input row
  struct Seg { int64_t fFlat, nFlat, pFlat, fGp, nGp, pGp, side; };
  Seg seg() const {
    Seg s{};
    int64_t o = 0;
    const int64_t ct = cfg.Ctop();
    if (cfg.flat) {
      s.fFlat = o; o += cfg.top(cfg.Lf) * ct;
      s.nFlat = o; o += cfg.top(cfg.Ln) * ct;
      s.pFlat = o; o += cfg.top(cfg.Lp) * ct;
    }
    s.fGp = o; o += 2 * ct; s.nGp = o; o += 2 * ct; s.pGp = o; o += 2 * ct;
    s.side = o;
    return s;
  }

  void forward(int B, bool train, real p, uint64_t seed) {
    reserve(B);
    sf.forward(scr, B);
    sl.forward(scr, 2 * B);
    const int64_t Dh = D(), ct = cfg.Ctop();
    const Seg g = seg();
    real* hin = Hin.data();
    if (cfg.flat) {
      copyRows(sf.top(), 0, cfg.top(cfg.Lf) * ct, hin + g.fFlat, Dh, B);
      copyRows(sl.top(), 0, cfg.top(cfg.Ln) * ct, hin + g.nFlat, Dh, B);
      copyRows(sl.top(), int64_t(B) * cfg.top(cfg.Ln) * ct, cfg.top(cfg.Lp) * ct, hin + g.pFlat, Dh, B);
    }
    gpool(sf.top(), hin + g.fGp, B, sf.Ltop(), int(ct), Dh);
    gpool(sl.top(), hin + g.nGp, B, sl.Ltop(), int(ct), Dh);
    gpool(sl.top() + int64_t(B) * sl.Ltop() * ct, hin + g.pGp, B, sl.Ltop(), int(ct), Dh);
    copyRows(side.data(), 0, N_SIDE, hin + g.side, Dh, B);
    const real* x = hin;
    for (size_t l = 0; l < head.size(); ++l) {
      bool last = l + 1 == head.size();
      real* y = last ? Zo.data() : HA[l]->data();
      head[l]->forward(x, y, B);
      if (!last) actDropout(y, HM[l]->data(), int64_t(B) * cfg.H[l], train, p, seed * 8 + 1 + l);
      x = y;
    }
    real* z = Zo.data(); real* pp = P.data();
    bk::parallel_for(int64_t(B) * NB, [=] BK_HD (int64_t i) { pp[i] = real(1) / (real(1) + rexp(-z[i])); });
  }
  // src rows of width w (row r at src + base + r*w) -> dst row r at dst + r*ds
  static void copyRows(const real* src, int64_t base, int64_t w, real* dst, int64_t ds, int B) {
    bk::parallel_for(int64_t(B) * w, [=] BK_HD (int64_t i) {
      int64_t r = i / w, c = i % w;
      dst[r * ds + c] = src[base + r * w + c];
    });
  }
  static void copyRowsBack(const real* src, int64_t ss, int64_t w, real* dst, int64_t base, int B) {
    bk::parallel_for(int64_t(B) * w, [=] BK_HD (int64_t i) {
      int64_t r = i / w, c = i % w;
      dst[base + r * w + c] = src[r * ss + c];
    });
  }
  static void actDropout(real* a, real* m, int64_t n, bool train, real p, uint64_t seed) {
    const real keep = 1.f - p;
    bk::parallel_for(n, [=] BK_HD (int64_t i) {
      real d = a[i] > 0.f ? 1.f : 0.f;
      if (train && p > 0.f) {
        uint64_t x = uint64_t(i) * 0x9E3779B97F4A7C15ull + seed * 0xD1B54A32D192ED03ull;
        x ^= x >> 33; x *= 0xFF51AFD7ED558CCDull; x ^= x >> 33;
        real u = real(x >> 40) * (1.f / 16777216.f);
        d = (u < keep) ? d / keep : 0.f;
      }
      m[i] = d;
      a[i] = a[i] * d;
    });
  }

  // mean over the batch of the BCE (posWeight, label smoothing ls) against T
  void backward(int B, real posWeight = 1.f, real ls = 0.f) {
    real* d = Dout.data(); const real* pp = P.data(); const real* t = T.data();
    const real invB = 1.f / real(B), pw = posWeight;
    const real* eg = extraGrad;
    bk::parallel_for(int64_t(B) * NB, [=] BK_HD (int64_t i) {
      real tt = t[i] * (1.f - ls) + 0.5f * ls;
      d[i] = (pp[i] * (pw * tt + 1.f - tt) - pw * tt) * invB + (eg ? eg[i] : 0.f);
    });
    extraGrad = nullptr;
    const real* dy = Dout.data();
    for (int l = int(head.size()) - 1; l >= 0; --l) {
      const real* x = l == 0 ? Hin.data() : HA[l - 1]->data();
      real* dx = l == 0 ? dHin.data() : HD[l - 1]->data();
      head[l]->backward(scr, x, dy, dx, B);
      if (l > 0) {
        const real* m = HM[l - 1]->data();
        bk::parallel_for(int64_t(B) * cfg.H[l - 1], [=] BK_HD (int64_t i) { dx[i] *= m[i]; });
      }
      dy = dx;
    }
    const int64_t Dh = D(), ct = cfg.Ctop();
    const Seg g = seg();
    const int64_t wf = cfg.top(cfg.Lf) * ct, wl = cfg.top(cfg.Ln) * ct;
    real* dtf = sf.dTop(); real* dtl = sl.dTop();
    if (cfg.flat) {
      copyRowsBack(dHin.data() + g.fFlat, Dh, wf, dtf, 0, B);
      copyRowsBack(dHin.data() + g.nFlat, Dh, wl, dtl, 0, B);
      copyRowsBack(dHin.data() + g.pFlat, Dh, wl, dtl, int64_t(B) * wl, B);
    } else {
      fill0(dtf, int64_t(B) * wf); fill0(dtl, int64_t(2 * B) * wl);
    }
    gpoolBackAdd(sf.top(), dHin.data() + g.fGp, dtf, B, sf.Ltop(), int(ct), Dh);
    gpoolBackAdd(sl.top(), dHin.data() + g.nGp, dtl, B, sl.Ltop(), int(ct), Dh);
    gpoolBackAdd(sl.top() + int64_t(B) * wl, dHin.data() + g.pGp, dtl + int64_t(B) * wl, B, sl.Ltop(), int(ct), Dh);
    sf.backward(scr, B);
    sl.backward(scr, 2 * B);
  }
  static void fill0(real* p, int64_t n) { bk::parallel_for(n, [=] BK_HD (int64_t i) { p[i] = 0.f; }); }

  // ---- file "CNN1" ----
  // int32 magic 0x314E4E43, version 1; Cfg: F, off(real), Lf, Ln, Lp, C0, k, nChans, chans[], flat,
  // nH, H[], maxPeaks, pairK; NB; params in params() order; rE, rF, nVocab, vocab[]; hasCal, a[], c[], q[]
  void save(const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    auto wi = [&](int32_t x) { f.write(reinterpret_cast<const char*>(&x), 4); };
    wi(0x314E4E43); wi(1);
    wi(cfg.F); f.write(reinterpret_cast<const char*>(&cfg.off), 4);
    wi(cfg.Lf); wi(cfg.Ln); wi(cfg.Lp); wi(cfg.C0); wi(cfg.k);
    wi(int32_t(cfg.chans.size())); for (int c : cfg.chans) wi(c);
    wi(cfg.flat);
    wi(int32_t(cfg.H.size())); for (int h : cfg.H) wi(h);
    wi(cfg.maxPeaks); wi(cfg.pairK); wi(NB);
    std::vector<real> h;
    for (Param* p : params()) { download(p->w, h, p->n); std::vector<float> hf(h.begin(), h.end()); f.write(reinterpret_cast<const char*>(hf.data()), 4 * hf.size()); }
    wi(rE); wi(rF); wi(int32_t(vocab.size()));
    if (!vocab.empty()) f.write(reinterpret_cast<const char*>(vocab.data()), 8 * vocab.size());
    bool hc = int(calA.size()) == NB;
    wi(hc ? 1 : 0);
    if (hc) {
      f.write(reinterpret_cast<const char*>(calA.data()), 4 * NB);
      f.write(reinterpret_cast<const char*>(calC.data()), 4 * NB);
      f.write(reinterpret_cast<const char*>(quality.data()), 4 * NB);
    }
  }
  static bool isCnnFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    int32_t m = 0; f.read(reinterpret_cast<char*>(&m), 4);
    return bool(f) && m == 0x314E4E43;
  }
  bool load(const std::string& path, bool train = false) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    auto ri = [&]() { int32_t x = 0; f.read(reinterpret_cast<char*>(&x), 4); return x; };
    if (ri() != 0x314E4E43) return false;
    ri();
    cfg.F = ri(); f.read(reinterpret_cast<char*>(&cfg.off), 4);
    cfg.Lf = ri(); cfg.Ln = ri(); cfg.Lp = ri(); cfg.C0 = ri(); cfg.k = ri();
    cfg.chans.assign(ri(), 0); for (int& c : cfg.chans) c = ri();
    cfg.flat = ri();
    cfg.H.assign(ri(), 0); for (int& h : cfg.H) h = ri();
    cfg.maxPeaks = ri(); cfg.pairK = ri(); NB = ri();
    build(1, train);
    for (Param* p : params()) {
      std::vector<float> hf(p->n);
      f.read(reinterpret_cast<char*>(hf.data()), 4 * hf.size()); std::vector<real> h(hf.begin(), hf.end());
      p->init(p->n, h, train);
    }
    rE = ri(); rF = ri();
    vocab.resize(ri());
    if (!vocab.empty()) f.read(reinterpret_cast<char*>(vocab.data()), 8 * vocab.size());
    int hc = ri();
    calA.clear(); calC.clear(); quality.clear();
    if (hc) {
      calA.resize(NB); calC.resize(NB); quality.resize(NB);
      f.read(reinterpret_cast<char*>(calA.data()), 4 * NB);
      f.read(reinterpret_cast<char*>(calC.data()), 4 * NB);
      f.read(reinterpret_cast<char*>(quality.data()), 4 * NB);
    }
    return bool(f);
  }
};

}  // namespace cnn
