// rev3_model.hpp: copy of rev/rev2_model.hpp (same file format) + Model::extraGrad
// (an extra dLoss/dlogit added in backward, for the within-window contrastive loss).
// Reverse model v2: spectrum -> fingerprint-bit probabilities, generalized
// from rev_model.hpp (which stays untouched - phys_ana and rev_train_ens
// depend on its file format):
//   - input layout is a runtime config (bin width, fragment / neutral-loss
//     ranges), stored in the model file
//   - any number of hidden layers (ReLU + dropout), any widths
//   - any number of output bits; the model file can carry the fp2
//     vocabulary its bits are defined by (nVocab 0 = folded ecfp.h r2 NB)
//   - optional per-bit Platt calibration (a, c) and a per-bit quality
//     (held-out F1) fitted on structure-disjoint held-out molecules
// Same hand-written backward as v1: first layer sparse (CSR fwd, CSC for
// dW1), rest dense via backend.hpp GEMMs.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "backend.hpp"

namespace rev2 {

constexpr int N_ADDUCT = 10, N_INSTR = 5;
struct Head { float precursor; int32_t mol, off, n, tag; };

constexpr float ADDUCT_SHIFT[10] = {
    1.007276f, -1.007276f, 22.989218f, 18.033823f, 38.963158f,
    34.969402f, 44.998201f, -17.002740f, 1.007276f, 22.989218f,
};

struct FeatCfg {
  float bin = 0.1f;      // Da per input bin
  int nbFrag = 10000;    // fragment bins (0 .. nbFrag*bin Da)
  int nbNL = 3000;       // neutral-loss bins
  int maxPeaks = 300;
  int offNL() const { return nbFrag; }
  int offAdduct() const { return nbFrag + nbNL; }
  int offInstr() const { return offAdduct() + N_ADDUCT; }
  int offMass() const { return offInstr() + N_INSTR; }
  int dIn() const { return offMass() + 1; }
};

inline void featurize(const FeatCfg& fc, const Head& h, const float* mz, const float* it,
                      std::vector<std::pair<int, float>>& row) {
  row.clear();
  int n = h.n;
  std::vector<int> idx(n);
  for (int i = 0; i < n; ++i) idx[i] = i;
  if (n > fc.maxPeaks) {
    std::nth_element(idx.begin(), idx.begin() + fc.maxPeaks, idx.end(),
                     [&](int a, int b) { return it[a] > it[b]; });
    idx.resize(fc.maxPeaks);
  }
  float mx = 0.f;
  for (int i : idx) mx = std::max(mx, it[i]);
  if (mx <= 0.f) mx = 1.f;
  for (int i : idx) {
    float v = std::sqrt(std::max(0.f, it[i]) / mx);
    int fb = int(mz[i] / fc.bin);
    if (fb >= 0 && fb < fc.nbFrag) row.emplace_back(fb, v);
    float nl = h.precursor - mz[i];
    int nb = int(nl / fc.bin);
    if (nl > 0.f && nb >= 0 && nb < fc.nbNL) row.emplace_back(fc.offNL() + nb, v);
  }
  int a = -1, s = -1;
  if (h.tag >= 0) {
    bool unkInstr = h.tag % 16 == 15;
    a = unkInstr ? h.tag / 16 + 1 : h.tag / 16;
    s = unkInstr ? -1 : h.tag % 16;
  }
  if (a >= 0 && a < N_ADDUCT) row.emplace_back(fc.offAdduct() + a, 1.f);
  if (s >= 0 && s < N_INSTR) row.emplace_back(fc.offInstr() + s, 1.f);
  row.emplace_back(fc.offMass(), h.precursor / 1000.f);
  std::sort(row.begin(), row.end());
  size_t w = 0;
  for (size_t r = 0; r < row.size(); ++r) {
    if (w > 0 && row[w - 1].first == row[r].first) row[w - 1].second = std::max(row[w - 1].second, row[r].second);
    else row[w++] = row[r];
  }
  row.resize(w);
}

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
  void init(int64_t count, const std::vector<float>& host, bool train = true) {
    n = count;
    upload(w, host);
    if (!train) return;
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

inline std::vector<float> randinit(int64_t n, float scale, uint64_t seed) {
  std::vector<float> h(n);
  uint64_t s = seed * 0x9E3779B97F4A7C15ull + 1;
  for (auto& x : h) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    float u1 = ((s >> 11) + 1) * (1.0f / 9007199254740993.0f);
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    float u2 = (s >> 11) * (1.0f / 9007199254740992.0f);
    x = std::sqrt(-2.f * std::log(u1)) * std::cos(6.2831853f * u2) * scale;
  }
  return h;
}

struct Model {
  FeatCfg fc;
  std::vector<int> H{3072, 3072};
  int NB = 2048;
  // fp2 vocabulary the output bits stand for (empty = folded ecfp.h r2 NB)
  int rE = -1, rF = -1;
  std::vector<uint64_t> vocab;
  // per-bit calibration (empty = none): p_cal = sigmoid(a * logit(p) + c)
  std::vector<float> calA, calC, quality;

  std::vector<std::unique_ptr<Param>> W, b;  // L+1 layers
  int maxB = 0;
  std::vector<std::unique_ptr<bk::Buffer<float>>> Z, M, A, DA, DZ;  // per hidden layer
  bk::Buffer<float> Zo, P, T, Dout;
  bk::Buffer<int32_t> rowptr, col, ccolptr, ccol, crow;
  bk::Buffer<float> val, cval;
  int nCols = 0;
  const float* extraGrad = nullptr;  // rev3: device pointer, B x NB, added to dLoss/dlogit

  int L() const { return int(H.size()); }
  int inDim(int l) const { return l == 0 ? fc.dIn() : H[l - 1]; }
  int outDim(int l) const { return l == L() ? NB : H[l]; }

  void alloc(bool train) {
    W.clear(); b.clear();
    for (int l = 0; l <= L(); ++l) { W.emplace_back(new Param()); b.emplace_back(new Param()); }
    Z.clear(); M.clear(); A.clear(); DA.clear(); DZ.clear();
    for (int l = 0; l < L(); ++l)
      for (auto* v : {&Z, &M, &A, &DA, &DZ}) v->emplace_back(new bk::Buffer<float>());
    maxB = 0;
    (void)train;
  }
  void init(uint64_t seed, float biasOut = -3.f) {
    alloc(true);
    for (int l = 0; l <= L(); ++l) {
      int64_t ni = inDim(l), no = outDim(l);
      float sc = l == 0 ? std::sqrt(2.f / 64.f) : (l == L() ? std::sqrt(1.f / ni) : std::sqrt(2.f / ni));
      W[l]->init(ni * no, randinit(ni * no, sc, seed + l));
      b[l]->init(no, std::vector<float>(no, l == L() ? biasOut : 0.f));
    }
  }
  void reserve(int B) {
    if (B <= maxB) return;
    maxB = B;
    for (int l = 0; l < L(); ++l)
      for (auto* v : {&Z, &M, &A, &DA, &DZ}) (*v)[l]->alloc(int64_t(B) * H[l]);
    for (auto* v : {&Zo, &P, &T, &Dout}) v->alloc(int64_t(B) * NB);
  }
  std::vector<Param*> params() {
    std::vector<Param*> p;
    for (int l = 0; l <= L(); ++l) { p.push_back(W[l].get()); p.push_back(b[l].get()); }
    return p;
  }

  // CSR of the batch; for training also the CSC (counting sort by column).
  void setBatch(const std::vector<std::vector<std::pair<int, float>>>& rows, bool withCsc) {
    std::vector<int32_t> rp(rows.size() + 1, 0), c;
    std::vector<float> v;
    size_t nnz = 0;
    for (auto& r : rows) nnz += r.size();
    c.reserve(nnz); v.reserve(nnz);
    for (size_t r = 0; r < rows.size(); ++r) {
      for (auto& e : rows[r]) { c.push_back(e.first); v.push_back(e.second); }
      rp[r + 1] = int32_t(c.size());
    }
    upload(rowptr, rp); upload(col, c); upload(val, v);
    if (!withCsc) return;
    const int D = fc.dIn();
    std::vector<int32_t> cnt(D + 1, 0);
    for (int32_t x : c) ++cnt[x + 1];
    for (int d = 0; d < D; ++d) cnt[d + 1] += cnt[d];
    std::vector<int32_t> cr(nnz);
    std::vector<float> cv(nnz);
    std::vector<int32_t> pos(cnt.begin(), cnt.end() - 1);
    for (size_t r = 0; r < rows.size(); ++r)
      for (int k = rp[r]; k < rp[r + 1]; ++k) {
        int32_t p = pos[c[k]]++;
        cr[p] = int32_t(r); cv[p] = v[k];
      }
    std::vector<int32_t> cp, cc;
    for (int d = 0; d < D; ++d)
      if (cnt[d + 1] > cnt[d]) { cc.push_back(d); cp.push_back(cnt[d]); }
    cp.push_back(int32_t(nnz));
    nCols = int(cc.size());
    upload(ccolptr, cp); upload(ccol, cc); upload(crow, cr); upload(cval, cv);
  }

  void forward(int B, bool train, float p, uint64_t seed) {
    reserve(B);
    {
      const int h1 = H[0];
      const int32_t* RP = rowptr.data(); const int32_t* C = col.data(); const float* V = val.data();
      const float* w1 = W[0]->w.data(); const float* bb1 = b[0]->w.data();
      float* z1 = Z[0]->data();
      bk::parallel_for(int64_t(B) * h1, [=] BK_HD (int64_t idx) {
        int64_t r = idx / h1, j = idx % h1;
        float s = bb1[j];
        for (int k = RP[r]; k < RP[r + 1]; ++k) s += V[k] * w1[int64_t(C[k]) * h1 + j];
        z1[idx] = s;
      });
      actDropout(*Z[0], *M[0], *A[0], int64_t(B) * h1, train, p, seed * 8 + 1);
    }
    for (int l = 1; l < L(); ++l) {
      bk::gemm(A[l - 1]->data(), W[l]->w.data(), Z[l]->data(), B, H[l - 1], H[l]);
      addBias(*Z[l], *b[l], B, H[l]);
      actDropout(*Z[l], *M[l], *A[l], int64_t(B) * H[l], train, p, seed * 8 + 1 + l);
    }
    bk::gemm(A[L() - 1]->data(), W[L()]->w.data(), Zo.data(), B, H[L() - 1], NB);
    addBias(Zo, *b[L()], B, NB);
    float* z3 = Zo.data(); float* pp = P.data();
    bk::parallel_for(int64_t(B) * NB, [=] BK_HD (int64_t i) { pp[i] = 1.f / (1.f + expf(-z3[i])); });
  }
  void addBias(bk::Buffer<float>& Zb, Param& bb, int B, int n) {
    float* z = Zb.data(); const float* bv = bb.w.data();
    bk::parallel_for(int64_t(B) * n, [=] BK_HD (int64_t i) { z[i] += bv[i % n]; });
  }
  void actDropout(bk::Buffer<float>& Zb, bk::Buffer<float>& Mb, bk::Buffer<float>& Ab,
                  int64_t n, bool train, float p, uint64_t seed) {
    const float* z = Zb.data(); float* m = Mb.data(); float* a = Ab.data();
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

  // Mean BCE against T (0/1); label smoothing ls moves targets to ls/2, 1-ls/2.
  void backward(int B, float posWeight, float ls) {
    const int nb = NB;
    float* d3 = Dout.data(); const float* pp = P.data(); const float* t = T.data();
    const float invB = 1.f / float(B), pw = posWeight;
    bk::parallel_for(int64_t(B) * nb, [=] BK_HD (int64_t i) {
      float tt = t[i] * (1.f - ls) + 0.5f * ls;
      d3[i] = (pp[i] * (pw * tt + 1.f - tt) - pw * tt) * invB;
    });
    if (extraGrad) {  // rev3: additional dLoss/dlogit (e.g. within-window contrastive term)
      const float* eg = extraGrad;
      bk::parallel_for(int64_t(B) * nb, [=] BK_HD (int64_t i) { d3[i] += eg[i]; });
    }
    bk::Buffer<float>* Dn = &Dout;
    for (int l = L(); l >= 1; --l) {
      int ni = H[l - 1], no = outDim(l);
      bk::gemm_tn(A[l - 1]->data(), Dn->data(), W[l]->g.data(), ni, no, B);
      colSum(*Dn, *b[l], B, no);
      bk::gemm_nt(Dn->data(), W[l]->w.data(), DA[l - 1]->data(), B, ni, no);
      mulInto(*DA[l - 1], *M[l - 1], *DZ[l - 1], int64_t(B) * ni);
      Dn = DZ[l - 1].get();
    }
    const int h1 = H[0];
    colSum(*DZ[0], *b[0], B, h1);
    fill(W[0]->g, W[0]->n, 0.f);
    const int32_t* CP = ccolptr.data(); const int32_t* CC = ccol.data();
    const int32_t* CR = crow.data(); const float* CV = cval.data();
    const float* dz1 = DZ[0]->data(); float* gw1 = W[0]->g.data();
    bk::parallel_for(int64_t(nCols) * h1, [=] BK_HD (int64_t idx) {
      int64_t u = idx / h1, j = idx % h1;
      float s = 0.f;
      for (int k = CP[u]; k < CP[u + 1]; ++k) s += CV[k] * dz1[int64_t(CR[k]) * h1 + j];
      gw1[int64_t(CC[u]) * h1 + j] = s;
    });
  }
  void colSum(bk::Buffer<float>& D, Param& bb, int B, int n) {
    const float* d = D.data(); float* g = bb.g.data();
    bk::parallel_for(n, [=] BK_HD (int64_t j) {
      float s = 0.f;
      for (int r = 0; r < B; ++r) s += d[int64_t(r) * n + j];
      g[j] = s;
    });
  }
  void mulInto(bk::Buffer<float>& Am, bk::Buffer<float>& Bm, bk::Buffer<float>& Cm, int64_t n) {
    const float* a = Am.data(); const float* bb = Bm.data(); float* c = Cm.data();
    bk::parallel_for(n, [=] BK_HD (int64_t i) { c[i] = a[i] * bb[i]; });
  }

  // ---- file format "REV2" ----
  //  int32 magic, version(1); float bin; int32 nbFrag, nbNL, maxPeaks;
  //  int32 L, H[L], NB; weights (W0 b0 W1 b1 ...); int32 rE, rF, nVocab,
  //  uint64 vocab[nVocab]; int32 hasCal; if hasCal: float a[NB], c[NB], q[NB]
  void save(const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    auto wi = [&](int32_t x) { f.write(reinterpret_cast<const char*>(&x), 4); };
    wi(0x32564552); wi(1);
    f.write(reinterpret_cast<const char*>(&fc.bin), 4);
    wi(fc.nbFrag); wi(fc.nbNL); wi(fc.maxPeaks);
    wi(L());
    for (int h : H) wi(h);
    wi(NB);
    std::vector<float> h;
    for (Param* p : params()) {
      download(p->w, h, p->n);
      f.write(reinterpret_cast<const char*>(h.data()), h.size() * 4);
    }
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
  // Loads REV2, or a v1 rev_model.hpp file (0x56455246, fixed 0.1 Da layout).
  bool load(const std::string& path, bool train = false) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    auto ri = [&]() { int32_t x = 0; f.read(reinterpret_cast<char*>(&x), 4); return x; };
    int32_t magic = ri();
    if (magic == 0x56455246) {
      fc = FeatCfg();
      if (ri() != fc.dIn()) return false;
      int h1 = ri(), h2 = ri();
      H = {h1, h2};
      NB = ri();
      rE = rF = -1; vocab.clear(); calA.clear(); calC.clear(); quality.clear();
    } else if (magic == 0x32564552) {
      ri();
      f.read(reinterpret_cast<char*>(&fc.bin), 4);
      fc.nbFrag = ri(); fc.nbNL = ri(); fc.maxPeaks = ri();
      int l = ri();
      H.assign(l, 0);
      for (int& h : H) h = ri();
      NB = ri();
    } else {
      return false;
    }
    alloc(train);
    for (int l = 0; l <= L(); ++l) {
      int64_t ni = inDim(l), no = outDim(l);
      std::vector<float> hw(ni * no), hb(no);
      f.read(reinterpret_cast<char*>(hw.data()), hw.size() * 4);
      f.read(reinterpret_cast<char*>(hb.data()), hb.size() * 4);
      W[l]->init(ni * no, hw, train);
      b[l]->init(no, hb, train);
    }
    if (magic == 0x32564552) {
      rE = ri(); rF = ri();
      int nv = ri();
      vocab.resize(nv);
      if (nv) f.read(reinterpret_cast<char*>(vocab.data()), 8 * size_t(nv));
      int hc = ri();
      calA.clear(); calC.clear(); quality.clear();
      if (hc) {
        calA.resize(NB); calC.resize(NB); quality.resize(NB);
        f.read(reinterpret_cast<char*>(calA.data()), 4 * NB);
        f.read(reinterpret_cast<char*>(calC.data()), 4 * NB);
        f.read(reinterpret_cast<char*>(quality.data()), 4 * NB);
      }
    }
    return bool(f);
  }
};

}  // namespace rev2
