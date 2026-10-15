// lm_feat.h - per-candidate feature vectors for the LambdaMART ranker (lmart.cpp / phys_lm.cpp).
//
// One function turns a query's raw per-candidate scores (the columns phys_lm
// dumps) into the ranker's features, so training (from the dump) and
// inference (inside phys_lm) go through exactly the same code.
//
// Raw columns per candidate (RAW order; the dump writes them in this order
// with inLib and the name between ana and g0, see phys_lm.cpp):
//   phys mlp rev lib ana | g0..g5 (generation: isGen, isGen*sim^3, isGen*2step,
//   isGen*log1p(nSup), isGen*log1p(nSib), isGen*opPrior) | massErr (ppm, best
//   window) | aCos aMcos aM6p2 aM6p4 aNP6 (other analog-propagation variants)
// Query meta: number of spectra, neutral mass, polarity, adduct, instrument.
//
// Deliberately NOT features (holdout-construction leaks, RESUME "作業の前提"):
// library membership (inLib), the pool's train flag, COCONUT membership or
// metadata, structural descriptors. lib (best cosine to the candidate's own
// library spectra) is 0 for a candidate without library spectra, so a tree
// can still split on lib > 0; the class-mix weights in lmart decide how much
// that is trusted.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <string>
#include <vector>

namespace lmf {

constexpr int NRAW = 17;
enum Raw { PHYS, MLP, REV, LIB, ANA, G0, G1, G2, G3, G4, G5, MERR, ACOS, AMCOS, AM6P2, AM6P4, ANP6 };
using Row = std::array<float, NRAW>;

struct QMeta {
  int nspec = 1;
  float mass = 0.f;   // neutral mass of the first spectrum's window
  int pol = 0;        // 1 negative, 0 positive, -1 unknown
  int adduct = -1;
  int instr = -1;     // tag % 16 (0 timsTOF, 1 Orbitrap, ...), -1 unknown
};

// Hand fusion (phys_gen recommended config, v20 + generation weights):
// z(phys) + 0.25 z(mlp) + 1.0 z(rev) + 8 lib^32 + 24 ana + gw . g
constexpr float HAND_W = 0.25f, HAND_WR = 1.f, HAND_WL = 8.f, HAND_POW = 32.f, HAND_WA = 24.f;
constexpr float HAND_GW[6] = {-2.f, 0.f, -1.f, 4.f, 0.f, 1.5f};

inline const std::vector<std::string>& names() {
  static const std::vector<std::string> n = {
      // 0-11: model scores, per query
      "z_phys", "rk_phys", "rf_phys", "gap_phys",
      "z_mlp", "rk_mlp", "rf_mlp", "gap_mlp",
      "z_rev", "rk_rev", "rf_rev", "gap_rev",
      // 12-16: fused
      "hand", "rk_hand", "gap_hand", "spec3", "rk_spec3",
      // 17-22: library identity
      "lib", "lib_m_max", "rk_lib", "q_maxlib", "q_lib2", "q_nlib80",
      // 23-31: analogs
      "ana", "ana_m_max", "rk_ana", "q_maxana", "a_cos", "a_mcos", "a_m6p2", "a_m6p4", "a_np6",
      // 32-37: generation
      "isGen", "g_sim3", "g_2step", "g_nsup", "g_nsib", "g_prior",
      // 38: mass error
      "merr",
      // 39-45: query
      "q_logn", "q_genfrac", "q_nspec", "q_mass", "q_pol", "q_adduct", "q_instr",
      // 46-47: raw model scores
      "raw_phys", "raw_rev"};
  return n;
}
inline int nf() { return int(names().size()); }

inline std::vector<float> zs(const std::vector<float>& v) {  // identical to phys_gen's zscore
  double m = 0, s = 0;
  for (float x : v) m += x;
  m /= double(std::max<size_t>(1, v.size()));
  for (float x : v) s += (x - m) * (x - m);
  s = std::sqrt(s / double(std::max<size_t>(1, v.size()))) + 1e-6;
  std::vector<float> z(v.size());
  for (size_t i = 0; i < v.size(); ++i) z[i] = float((v[i] - m) / s);
  return z;
}
// rank (0 = best) with ties sharing the best rank
inline std::vector<float> ranks(const std::vector<float>& v) {
  std::vector<int> o(v.size());
  std::iota(o.begin(), o.end(), 0);
  std::stable_sort(o.begin(), o.end(), [&](int a, int b) { return v[a] > v[b]; });
  std::vector<float> r(v.size());
  for (size_t k = 0; k < o.size(); ++k)
    r[o[k]] = (k > 0 && v[o[k]] == v[o[k - 1]]) ? r[o[k - 1]] : float(k);
  return r;
}

inline float handScore(float zp, float zm, float zr, float lib, float ana, const float* g) {
  float f = zp + HAND_W * zm + HAND_WR * zr + HAND_WL * std::pow(std::max(0.f, lib), HAND_POW) + HAND_WA * ana;
  for (int k = 0; k < 6; ++k) f += HAND_GW[k] * g[k];
  return f;
}

// X: n * nf() row-major
inline void build(const std::vector<Row>& R, const QMeta& m, std::vector<float>& X) {
  const size_t n = R.size();
  const int F = nf();
  X.assign(n * size_t(F), 0.f);
  if (n == 0) return;
  auto col = [&](int c) { std::vector<float> v(n); for (size_t i = 0; i < n; ++i) v[i] = R[i][c]; return v; };
  const float rn = float(std::max<size_t>(1, n - 1));
  std::vector<float> z[3], rk[3];
  int src[3] = {PHYS, MLP, REV};
  for (int s = 0; s < 3; ++s) {
    z[s] = zs(col(src[s]));
    rk[s] = ranks(z[s]);
    float zmax = *std::max_element(z[s].begin(), z[s].end());
    for (size_t i = 0; i < n; ++i) {
      float* x = &X[i * F + 4 * s];
      x[0] = z[s][i]; x[1] = std::min(rk[s][i], 50.f); x[2] = rk[s][i] / rn; x[3] = z[s][i] - zmax;
    }
  }
  std::vector<float> hand(n), spec(n);
  for (size_t i = 0; i < n; ++i) {
    hand[i] = handScore(z[0][i], z[1][i], z[2][i], R[i][LIB], R[i][ANA], &R[i][G0]);
    spec[i] = z[0][i] + HAND_W * z[1][i] + HAND_WR * z[2][i];
  }
  auto rh = ranks(hand), rsp = ranks(spec);
  float hmax = *std::max_element(hand.begin(), hand.end());
  auto lib = col(LIB), ana = col(ANA);
  auto rl = ranks(lib), ra = ranks(ana);
  std::vector<float> ls = lib;
  std::sort(ls.begin(), ls.end(), std::greater<float>());
  float lmax = ls[0], l2 = n > 1 ? ls[1] : 0.f;
  int nl80 = 0;
  for (float v : lib) nl80 += v >= 0.8f;
  float amax = *std::max_element(ana.begin(), ana.end());
  int ngen = 0;
  for (auto& r : R) ngen += r[G0] > 0.5f;
  for (size_t i = 0; i < n; ++i) {
    float* x = &X[i * F];
    x[12] = hand[i]; x[13] = std::min(rh[i], 50.f); x[14] = hand[i] - hmax;
    x[15] = spec[i]; x[16] = std::min(rsp[i], 50.f);
    x[17] = lib[i]; x[18] = lib[i] - lmax; x[19] = std::min(rl[i], 50.f); x[20] = lmax; x[21] = l2;
    x[22] = float(nl80);
    x[23] = ana[i]; x[24] = ana[i] - amax; x[25] = std::min(ra[i], 50.f); x[26] = amax;
    x[27] = R[i][ACOS]; x[28] = R[i][AMCOS]; x[29] = R[i][AM6P2]; x[30] = R[i][AM6P4]; x[31] = R[i][ANP6];
    for (int k = 0; k < 6; ++k) x[32 + k] = R[i][G0 + k];
    x[38] = std::min(R[i][MERR], 50.f);
    x[39] = std::log(float(n)); x[40] = float(ngen) / float(n); x[41] = float(m.nspec); x[42] = m.mass;
    x[43] = float(m.pol); x[44] = float(m.adduct); x[45] = float(m.instr);
    x[46] = R[i][PHYS]; x[47] = R[i][REV];
  }
}

}  // namespace lmf
