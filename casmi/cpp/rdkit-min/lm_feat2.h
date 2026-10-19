// lm_feat2.h - features of the v29 LambdaMART ranker (lmart_v29.cpp trainer,
// phys_v29lm.cpp inference). Same idea as lm_feat.h (one function for training
// and inference), for the v27a/v28a base:
//   * phys = fragment-size physics (phys_size_fqry.txt), lib capped at 1 (libcap=1),
//     rev = REV2 (vocabulary bits) - as production v28a
//   * + pool popularity (pop = PubChem log1p(#SID)+log1p(#PMID), 0 for generated)
//   * hand = the v28a hand fusion before FPNet / ppop
// Raw columns (dump order after "phys mlp rev lib ana inLib name"):
//   g0..g5 merr aCos aMcos aM6p2 aM6p4 aNP6 pop | query: nspec mass pol adduct instr
//
// Not features, ever (holdout-construction leaks): inLib, the pool's train flag,
// COCONUT membership/metadata. Query meta (nspec, mass, pol, adduct, instr) and the
// raw (unnormalised) model scores are built but meant to be dropped (drop= in the
// trainer): the holdouts differ in them by construction, and training uses
// holdout-safe models whose raw scale differs from the production models.
//
// Option libfloor (Opt::libFloor): lib values below it are set to 0 before any
// feature is built, so "has library spectra but they do not match" looks exactly
// like "has no library spectra" - the membership shortcut the denovo holdouts teach.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <string>
#include <vector>

namespace lmf2 {

constexpr int NRAW = 18;
enum Raw { PHYS, MLP, REV, LIB, ANA, G0, G1, G2, G3, G4, G5, MERR, ACOS, AMCOS, AM6P2, AM6P4, ANP6, POP };
using Row = std::array<float, NRAW>;

struct QMeta {
  int nspec = 1;
  float mass = 0.f;
  int pol = 0;
  int adduct = -1;
  int instr = -1;
};

struct Opt {
  float libFloor = 0.f;
};

// v28a hand fusion (before FPNet and pool popularity):
// z(phys) + 0.25 z(mlp) + 1.0 z(rev) + 8 lib^32 + 24 ana + gw . g   (lib already capped at 1)
constexpr float HAND_W = 0.25f, HAND_WR = 1.f, HAND_WL = 8.f, HAND_POW = 32.f, HAND_WA = 24.f;
constexpr float HAND_GW[6] = {-2.f, 0.f, -1.f, 4.f, 0.f, 1.5f};

inline const std::vector<std::string>& names() {
  static const std::vector<std::string> n = {
      "z_phys", "rk_phys", "rf_phys", "gap_phys",                    // 0-3
      "z_mlp", "rk_mlp", "rf_mlp", "gap_mlp",                        // 4-7
      "z_rev", "rk_rev", "rf_rev", "gap_rev",                        // 8-11
      "hand", "rk_hand", "gap_hand", "z_hand", "rf_hand", "spec3", "rk_spec3",  // 12-18
      "lib", "lib_m_max", "rk_lib", "q_maxlib", "q_lib2", "q_nlib80",           // 19-24
      "ana", "ana_m_max", "rk_ana", "q_maxana", "a_cos", "a_mcos", "a_m6p2", "a_m6p4", "a_np6",  // 25-33
      "isGen", "g_sim3", "g_2step", "g_nsup", "g_nsib", "g_prior",   // 34-39
      "merr",                                                        // 40
      "q_logn", "q_genfrac",                                         // 41-42
      "pop", "z_pop", "rk_pop", "pop_m_max",                         // 43-46
      "q_nspec", "q_mass", "q_pol", "q_adduct", "q_instr",           // 47-51
      "raw_phys", "raw_rev"};                                        // 52-53
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
inline std::vector<float> ranks(const std::vector<float>& v) {  // 0 = best, ties share the best rank
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

// hand fusion of a query from raw rows (no floor: the production hand score)
inline std::vector<float> handOf(const std::vector<Row>& R) {
  const size_t n = R.size();
  std::vector<float> p(n), m(n), r(n), h(n);
  for (size_t i = 0; i < n; ++i) { p[i] = R[i][PHYS]; m[i] = R[i][MLP]; r[i] = R[i][REV]; }
  auto zp = zs(p), zm = zs(m), zr = zs(r);
  for (size_t i = 0; i < n; ++i) h[i] = handScore(zp[i], zm[i], zr[i], R[i][LIB], R[i][ANA], &R[i][G0]);
  return h;
}

// X: n * nf() row-major
inline void build(std::vector<Row> R, const QMeta& m, std::vector<float>& X, const Opt& opt = Opt()) {
  const size_t n = R.size();
  const int F = nf();
  X.assign(n * size_t(F), 0.f);
  if (n == 0) return;
  const std::vector<float> hand = handOf(R);  // before the floor: the production hand score
  if (opt.libFloor > 0.f)
    for (auto& r : R) if (r[LIB] < opt.libFloor) r[LIB] = 0.f;
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
  std::vector<float> spec(n);
  for (size_t i = 0; i < n; ++i) spec[i] = z[0][i] + HAND_W * z[1][i] + HAND_WR * z[2][i];
  auto rh = ranks(hand), rsp = ranks(spec), zh = zs(hand);
  float hmax = *std::max_element(hand.begin(), hand.end());
  auto lib = col(LIB), ana = col(ANA), pop = col(POP);
  auto rl = ranks(lib), ra = ranks(ana);
  std::vector<float> ls = lib;
  std::sort(ls.begin(), ls.end(), std::greater<float>());
  float lmax = ls[0], l2 = n > 1 ? ls[1] : 0.f;
  int nl80 = 0;
  for (float v : lib) nl80 += v >= 0.8f;
  float amax = *std::max_element(ana.begin(), ana.end());
  int ngen = 0;
  for (auto& r : R) ngen += r[G0] > 0.5f;
  // popularity over pool candidates (generated: pop 0, z = the pool minimum)
  std::vector<float> pp;
  for (size_t i = 0; i < n; ++i) if (R[i][G0] < 0.5f) pp.push_back(pop[i]);
  double pm = 0, psd = 0, pmax = 0, pmin = 0;
  if (!pp.empty()) {
    for (float v : pp) pm += v;
    pm /= pp.size();
    for (float v : pp) psd += (v - pm) * (v - pm);
    psd = std::sqrt(psd / pp.size()) + 1e-6;
    pmax = *std::max_element(pp.begin(), pp.end());
    pmin = *std::min_element(pp.begin(), pp.end());
  } else psd = 1;
  std::vector<float> popR = pop;
  for (size_t i = 0; i < n; ++i) if (R[i][G0] >= 0.5f) popR[i] = -1.f;
  auto rp = ranks(popR);
  for (size_t i = 0; i < n; ++i) {
    float* x = &X[i * F];
    x[12] = hand[i]; x[13] = std::min(rh[i], 50.f); x[14] = hand[i] - hmax; x[15] = zh[i]; x[16] = rh[i] / rn;
    x[17] = spec[i]; x[18] = std::min(rsp[i], 50.f);
    x[19] = lib[i]; x[20] = lib[i] - lmax; x[21] = std::min(rl[i], 50.f); x[22] = lmax; x[23] = l2;
    x[24] = float(nl80);
    x[25] = ana[i]; x[26] = ana[i] - amax; x[27] = std::min(ra[i], 50.f); x[28] = amax;
    x[29] = R[i][ACOS]; x[30] = R[i][AMCOS]; x[31] = R[i][AM6P2]; x[32] = R[i][AM6P4]; x[33] = R[i][ANP6];
    for (int k = 0; k < 6; ++k) x[34 + k] = R[i][G0 + k];
    x[40] = std::min(R[i][MERR], 50.f);
    x[41] = std::log(float(n)); x[42] = float(ngen) / float(n);
    const bool gen = R[i][G0] >= 0.5f;
    x[43] = gen ? 0.f : pop[i];
    x[44] = float(((gen ? pmin : pop[i]) - pm) / psd);
    x[45] = std::min(rp[i], 50.f);
    x[46] = float((gen ? pmin : pop[i]) - pmax);
    x[47] = float(m.nspec); x[48] = m.mass; x[49] = float(m.pol); x[50] = float(m.adduct); x[51] = float(m.instr);
    x[52] = R[i][PHYS]; x[53] = R[i][REV];
  }
}

}  // namespace lmf2
