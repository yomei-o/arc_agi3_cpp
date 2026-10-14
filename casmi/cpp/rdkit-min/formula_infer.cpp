// Molecular formula posterior from MS/MS spectra, by chemistry and
// arithmetic plus a three-number mass-defect model.
//
// Candidates. The neutral mass M = precursor - adduct. Candidate formulas F
// come from (a) enumerating every CcHhNnOoPpSsFfClxBry within +-ppm of M
// that passes integer-RDBE and Kind & Fiehn's element-ratio rules, and/or
// (b) the distinct formulas of the structure pool (pool730) in the window.
//
// Likelihood. Every product ion is a sub-formula f of the precursor ion
// I = F + adduct, with 2*RDBE(f) >= -1. Per peak i (m/z m_i, weight w_i):
//
//   LR_i(F) = (1-pi) + pi * sum_{f in S(F), |m_f-m_i|<4s} phi(m_i-m_f; s_i)
//                          / ( N_F(M_f) * q(m_i - M_f | M_f) )
//
// N_F(M) = number of sub-formulas of I at nominal mass M (the Occam factor:
// a richer element set spreads its probability over more sub-formulas), q
// the mass-defect density of real fragment peaks (Gaussian per 50-Da bin,
// fitted by `fitq` on fqry peaks explained by true formulas, mixed with 5%
// uniform), w_i = sqrt(I_i)/max sqrt(I), top 30 peaks below precursor-0.5.
// log P(F|D) = log prior(F) + T * sum_spectra sum_i w_i log LR_i(F),
// normalised over the candidates (T=0.5 calibrated on fqry).
//
// Enumerated candidates use an alphabet prior: at most one halogen type and
// at most 4 S+P+halogen atoms (~99% of pool730 and of the holdout truths;
// without it a 1 kDa precursor has >20k candidates and takes minutes).
//
// Ablation on fqry (sigma 3 ppm / 0.5 mDa): dropping 1/N_F(M) takes the
// enumerated top-1 from 42.7% to 10.8%; the q model is worth ~1 point.
//
// For reranking STRUCTURES, use a uniform-prior variant (U/A/B0): the pool
// count prior counts each structure's formula once per isomer, which is
// right for P(formula) but double-counts when every structure is scored.
//
// N_F(M) by the generating function prod_e sum_{k=0}^{n_e} z^{k m_e} with a
// second variable y^{k s_e} tracking 2*RDBE (s_C=2, s_N=s_P=1, s_O=s_S=0,
// halogens/Na/K -1, H -1), i.e. the bounded-knapsack difference equation
//   N_k(M,r) = N_{k-1}(M,r) + N_k(M-m_k, r-s_k) - N_{k-1}(M-(n_k+1)m_k, r-(n_k+1)s_k)
// over the non-H elements, then H closed-form via suffix sums over r
// (2*RDBE = r - h + 2 >= -1) - default (rdbecount=1). rdbecount=0 uses the
// plain 1-D G(z) (no RDBE condition; 10x faster, ~1 point worse); countAt()
// is a direct per-mass count, verified equal to the DP by `selftest`.
// The exact-mass matches S(F) near a peak are enumerated directly (heavy
// atoms recursively with mass-range pruning, C and H solved).
//
// Usage:
//   formula_infer poolcache <dir> <pool_stem> <out_formulas.txt>
//   formula_infer fitq <dir> <qry> [key=value...]
//   formula_infer eval <dir> <qry> [key=value...]
// keys (defaults): ppm=10 fppm=3 fabs=0.0005 pi=0.5 maxpk=30 q=a,b,c,d outT=0.5 rdbecount=1
//       poolf=<formulas cache> pool=<pool_stem> out=<file> outvar=B outT=1 alpha=1
//       qfile=<defect table from fitq> rdbecount=0|1 occam=0|1 qunif=0.05
// outvar: A (enumerated, uniform), B (pool formulas, prior = #structures),
// B0 (pool formulas, uniform), C (union, log(count+alpha)), U (union, uniform).
#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <GraphMol/GraphMol.h>
#include <GraphMol/PeriodicTable.h>
#include <GraphMol/SmilesParse/SmilesParse.h>

namespace {

// Element order is fixed throughout: C H N O P S F Cl Br Na K.
constexpr int NE = 11;
enum { C, H, N, O, P, S, F, Cl, Br, Na, K };
constexpr double MASS[NE] = {12.0, 1.00782503207, 14.0030740048, 15.99491461956,
                             30.97376163, 31.97207100, 18.99840322, 34.96885268,
                             78.9183371, 22.9897692809, 38.96370668};
constexpr int NOM[NE] = {12, 1, 14, 16, 31, 32, 19, 35, 79, 23, 39};
constexpr int RSTEP[NE] = {2, -1, 1, 0, 1, 0, -1, -1, -1, -1, -1};  // contribution to 2*RDBE-2
constexpr const char* SYM[NE] = {"C", "H", "N", "O", "P", "S", "F", "Cl", "Br", "Na", "K"};
constexpr double ELECTRON = 0.00054857990946;
// Hill order: C, H, then alphabetical.
constexpr int HILL[NE] = {C, H, Br, Cl, F, K, N, Na, O, P, S};

using Formula = std::array<int, NE>;

double massOf(const Formula& f) {
  double m = 0;
  for (int e = 0; e < NE; ++e) m += f[e] * MASS[e];
  return m;
}
int nomOf(const Formula& f) {
  int m = 0;
  for (int e = 0; e < NE; ++e) m += f[e] * NOM[e];
  return m;
}
int twoRdbe(const Formula& f) {
  int r = 2;
  for (int e = 0; e < NE; ++e) r += f[e] * RSTEP[e];
  return r;
}

std::string hill(const Formula& f) {
  std::string s;
  for (int k = 0; k < NE; ++k) {
    int e = HILL[k];
    if (f[e]) s += SYM[e] + (f[e] > 1 ? std::to_string(f[e]) : "");
  }
  return s;
}

bool parseHill(const std::string& s, Formula& f) {
  f.fill(0);
  size_t i = 0;
  while (i < s.size()) {
    if (!std::isupper((unsigned char)s[i])) return false;
    std::string sym(1, s[i++]);
    if (i < s.size() && std::islower((unsigned char)s[i])) sym += s[i++];
    int n = 0;
    bool any = false;
    while (i < s.size() && std::isdigit((unsigned char)s[i])) { n = n * 10 + (s[i++] - '0'); any = true; }
    if (!any) n = 1;
    int e = -1;
    for (int k = 0; k < NE; ++k) if (sym == SYM[k]) e = k;
    if (e < 0) return false;
    f[e] += n;
  }
  return true;
}

// Adduct vocabulary of make_holdout.py; ion = M + delta, charge z.
struct Adduct { Formula delta; int z; bool ok; };
Adduct adductOf(int a) {
  Adduct d{}; d.delta.fill(0); d.ok = true; d.z = 0;
  switch (a) {
    case 0: d.delta[H] = 1; d.z = 1; break;                       // [M+H]+
    case 1: d.delta[H] = -1; d.z = -1; break;                     // [M-H]-
    case 2: d.delta[Na] = 1; d.z = 1; break;                      // [M+Na]+
    case 3: d.delta[N] = 1; d.delta[H] = 4; d.z = 1; break;       // [M+NH4]+
    case 4: d.delta[K] = 1; d.z = 1; break;                       // [M+K]+
    case 5: d.delta[Cl] = 1; d.z = -1; break;                     // [M+Cl]-
    case 6: d.delta[C] = 1; d.delta[H] = 1; d.delta[O] = 2; d.z = -1; break;  // [M+CH2O2-H]-
    case 7: d.delta[H] = -1; d.delta[O] = -1; d.z = 1; break;     // [M+H-H2O]+
    default: d.ok = false;                                        // 2M adducts, unknown
  }
  return d;
}

// Kind & Fiehn element-ratio limits (the "common range" column).
bool plausible(const Formula& f) {
  if (f[C] < 1) return false;
  int r2 = twoRdbe(f);
  if (r2 < 0 || (r2 & 1)) return false;  // neutral: integer, non-negative RDBE
  double c = f[C];
  if (f[H] > 3.1 * c + 3 || f[H] < 0.1 * c - 1) return false;
  if (f[N] > 1.3 * c || f[O] > 1.2 * c + 1 || f[P] > 0.3 * c + 1 || f[S] > 0.8 * c + 1) return false;
  if (f[F] > 1.5 * c || f[Cl] > 0.8 * c || f[Br] > 0.8 * c) return false;
  int atoms = 0;
  for (int e = 0; e < NE; ++e) atoms += f[e];
  int val = 4 * f[C] + f[H] + 3 * f[N] + 2 * f[O] + 3 * f[P] + 2 * f[S] + f[F] + f[Cl] + f[Br];
  return val >= 2 * (atoms - 1);
}

std::vector<Formula> enumerate(double M, double ppm) {
  std::vector<Formula> out;
  double tol = M * ppm * 1e-6;
  int maxC = int(M / 12.0) + 1;
  Formula f; f.fill(0);
  for (int br = 0; br <= 3; ++br)
   for (int cl = 0; cl <= 4; ++cl)
    for (int fl = 0; fl <= 6; ++fl)
     for (int s = 0; s <= 3; ++s)
      for (int p = 0; p <= 2; ++p)
       for (int n = 0; n <= 12; ++n)
        for (int o = 0; o <= 40; ++o) {
          // Alphabet prior (covers ~99% of pool730 and of the holdout truths):
          // at most one halogen type, at most 4 S+P+halogen atoms.
          if ((br > 0) + (cl > 0) + (fl > 0) > 1 || br + cl + fl + s + p > 4) break;
          double base = br * MASS[Br] + cl * MASS[Cl] + fl * MASS[F] + s * MASS[S] + p * MASS[P]
                        + n * MASS[N] + o * MASS[O];
          if (base > M + tol) break;
          for (int c = 1; c <= maxC; ++c) {
            double rest = M - base - c * 12.0;
            if (rest < -tol) break;
            int h = int(std::lround(rest / MASS[H]));
            for (int hh = std::max(0, h - 1); hh <= h + 1; ++hh) {
              double m = base + c * 12.0 + hh * MASS[H];
              if (std::fabs(m - M) > tol) continue;
              f = {c, hh, n, o, p, s, fl, cl, br, 0, 0};
              if (plausible(f)) out.push_back(f);
            }
          }
        }
  return out;
}

struct Head { float precursor; int32_t mol, off, n, tag; };

std::vector<char> slurp(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) { std::fprintf(stderr, "cannot read %s\n", path.c_str()); std::exit(1); }
  std::streamsize n = f.tellg();
  f.seekg(0);
  std::vector<char> buf((size_t(n)));
  f.read(buf.data(), n);
  return buf;
}

int adductIdx(int32_t tag) {
  if (tag < 0) return -1;
  return tag % 16 == 15 ? tag / 16 + 1 : tag / 16;
}

Formula formulaOfSmiles(const std::string& smi, bool& ok) {
  Formula f; f.fill(0); ok = false;
  std::unique_ptr<RDKit::RWMol> m;
  try { m.reset(RDKit::SmilesToMol(smi)); } catch (...) { return f; }
  if (!m) return f;
  for (auto atom : m->atoms()) {
    std::string s = atom->getSymbol();
    int e = -1;
    for (int k = 0; k < NE; ++k) if (s == SYM[k]) e = k;
    if (e < 0) return f;  // element outside the enumerated set
    f[e] += 1;
    f[H] += int(atom->getTotalNumHs());
    if (atom->getFormalCharge() != 0) return f;  // permanent ions: skip
  }
  ok = true;
  return f;
}

// ---------------------------------------------------------------------------
// Sub-formula counting: N(M) = #{f <= I : nominal(f) = M, 2RDBE(f) >= -1}.
struct SubCounter {
  int Mmax = 0, rmin = 0, rmax = 0, W = 1, nH = 0;
  std::vector<double> S;  // suffix sums over r: S[M*W + (r-rmin)] = sum_{r'>=r} D[M][r']
  void build(const Formula& ion) {
    Formula noH = ion; noH[H] = 0;
    nH = ion[H];
    Mmax = nomOf(noH) + 1;
    rmin = 0; rmax = 0;
    for (int e = 0; e < NE; ++e) {
      if (e == H) continue;
      if (RSTEP[e] < 0) rmin += RSTEP[e] * ion[e]; else rmax += RSTEP[e] * ion[e];
    }
    W = rmax - rmin + 1;
    size_t sz = size_t(Mmax + 1) * W;
    std::vector<double> D(sz, 0.0), E(sz);
    D[0 * W + (0 - rmin)] = 1.0;
    for (int e = 0; e < NE; ++e) {
      if (e == H || ion[e] == 0) continue;
      int m = NOM[e], s = RSTEP[e], n = ion[e];
      int m2 = (n + 1) * m, s2 = (n + 1) * s;
      for (int M = 0; M <= Mmax; ++M) {
        double* Er = &E[size_t(M) * W];
        const double* Dr = &D[size_t(M) * W];
        const double* E1 = M >= m ? &E[size_t(M - m) * W] : nullptr;
        const double* D2 = M >= m2 ? &D[size_t(M - m2) * W] : nullptr;
        for (int ri = 0; ri < W; ++ri) {
          double v = Dr[ri];
          int r1 = ri - s, r2i = ri - s2;
          if (E1 && r1 >= 0 && r1 < W) v += E1[r1];
          if (D2 && r2i >= 0 && r2i < W) v -= D2[r2i];
          Er[ri] = v;
        }
      }
      D.swap(E);
    }
    S.assign(sz, 0.0);
    for (int M = 0; M <= Mmax; ++M) {
      double acc = 0;
      for (int ri = W - 1; ri >= 0; --ri) { acc += D[size_t(M) * W + ri]; S[size_t(M) * W + ri] = acc; }
    }
  }
  // Total nominal mass M of the ion-sub-formula (H included).
  double count(int M) const {
    double n = 0;
    for (int h = 0; h <= nH && h <= M; ++h) {
      int Mh = M - h;
      if (Mh > Mmax) continue;
      int r0 = h - 3;  // r - h + 2 >= -1
      if (r0 > rmax) break;
      int ri = std::max(r0, rmin) - rmin;
      n += S[size_t(Mh) * W + ri];
    }
    return n;
  }
};

// Exact-mass sub-formulas of `ion` (charge z) within +-tol of m/z `mz`.
struct Match { double mass; int nom; Formula g; };
const int HEAVY[] = {Br, Cl, K, Na, S, P, F, N, O};
void matchRec(const Formula& ion, int z, double target, double tol, int k, Formula& g, double part,
              std::vector<Match>& out, const double* maxRest) {
  if (part + maxRest[k] < target - tol) return;  // even all remaining atoms fall short
  if (k == 9) {
    // C and H: 12c + mH*h = target - part, 0 <= h <= nH, 0 <= c <= nC.
    double rest = target - part;
    if (rest < -tol) return;
    int cHi = std::min(ion[C], int(std::floor((rest + tol) / 12.0)));
    int cLo = std::max(0, int(std::ceil((rest - tol - ion[H] * MASS[H]) / 12.0)));
    for (int c = cLo; c <= cHi; ++c) {
      double r2 = rest - 12.0 * c;
      int h = int(std::lround(r2 / MASS[H]));
      if (h < 0 || h > ion[H]) continue;
      double m = part + 12.0 * c + h * MASS[H];
      if (std::fabs(m - target) > tol) continue;
      g[C] = c; g[H] = h;
      bool heavy = c > 0;
      for (int q = 0; q < 9 && !heavy; ++q) heavy = g[HEAVY[q]] > 0;
      if (heavy && twoRdbe(g) >= -1) out.push_back({m - z * ELECTRON, nomOf(g), g});
    }
    g[C] = 0; g[H] = 0;
    return;
  }
  int e = HEAVY[k];
  for (int n = 0; n <= ion[e]; ++n) {
    double p = part + n * MASS[e];
    if (p > target + tol) break;
    g[e] = n;
    matchRec(ion, z, target, tol, k + 1, g, p, out, maxRest);
  }
  g[e] = 0;
}
void matchAll(const Formula& ion, int z, double target, double tol, std::vector<Match>& out) {
  double maxRest[10];
  maxRest[9] = ion[C] * 12.0 + ion[H] * MASS[H];
  for (int k = 8; k >= 0; --k) maxRest[k] = maxRest[k + 1] + ion[HEAVY[k]] * MASS[HEAVY[k]];
  Formula g; g.fill(0);
  matchRec(ion, z, target, tol, 0, g, 0.0, out, maxRest);
}

struct Peak { double mz, w, sig; };
struct Spec { int adduct; std::vector<Peak> pk; };

struct Params {
  double ppm = 10, fppm = 3, fabs = 0.0005, pi = 0.5;  // sigma tuned on fqry
  int maxpk = 30;
  double qa = 0.0, qb = 0.00045, qc = 0.02, qd = 0.0002;  // defect mean a+bM, sd c+dM
  double qunif = 0.05;
  double alpha = 1.0;
  std::string poolf, pool = "pool730", out, outvar = "B", qfile;
  std::vector<std::array<double, 3>> qtab;  // (M center, mean, sd), 50-Da bins
  double outT = 0.5;  // calibrated on fqry (best mean log p(truth))
  int dpcount = 0;
  double prune = 40;
  int rdbecount = 1;
  int occam = 1;
};

double qdens(const Params& P, int M, double d) {
  double mu = P.qa + P.qb * M, sd = P.qc + P.qd * M;
  if (!P.qtab.empty()) {
    const auto& t = P.qtab;
    if (M <= t.front()[0]) { mu = t.front()[1]; sd = t.front()[2]; }
    else if (M >= t.back()[0]) { mu = t.back()[1]; sd = t.back()[2]; }
    else {
      size_t i = size_t((M - t.front()[0]) / (t[1][0] - t[0][0]));
      double f = (M - t[i][0]) / (t[i + 1][0] - t[i][0]);
      mu = t[i][1] + f * (t[i + 1][1] - t[i][1]);
      sd = t[i][2] + f * (t[i + 1][2] - t[i][2]);
    }
  }
  sd = std::max(5e-3, sd);
  double z = (d - mu) / sd;
  return (1 - P.qunif) * std::exp(-0.5 * z * z) / (sd * 2.5066282746) + P.qunif * 1.0;
}

// N(M) at a single nominal mass, directly: the same generating-function
// coefficient as SubCounter, but summed over heavy-atom combinations with
// the C/H part in closed form, so only the masses that a peak actually
// matched are ever counted (the full DP table costs ~1M cells per
// candidate at 1 kDa; this costs about as much as the matching itself).
double countRec(const Formula& ion, int M, int k, int partNom, int partR, bool heavy, const int* maxRestNom) {
  if (partNom > M || partNom + maxRestNom[k] < M) return 0;
  if (k == 9) {
    int R = M - partNom;  // = 12c + h
    // 0<=h=R-12c<=nH ; 0<=c<=nC ; 2RDBE = 2c - h + partR + 2 >= -1  <=>  14c >= R - partR - 3
    int cLo = 0;
    if (R - ion[H] > 0) cLo = (R - ion[H] + 11) / 12;
    int need = R - partR - 3;
    if (need > 0) cLo = std::max(cLo, (need + 13) / 14);
    if (!heavy) cLo = std::max(cLo, 1);
    int cHi = std::min(ion[C], R / 12);
    return cHi >= cLo ? double(cHi - cLo + 1) : 0.0;
  }
  int e = HEAVY[k];
  double n = 0;
  for (int c = 0; c <= ion[e]; ++c) {
    int pn = partNom + c * NOM[e];
    if (pn > M) break;
    n += countRec(ion, M, k + 1, pn, partR + c * RSTEP[e], heavy || c > 0, maxRestNom);
  }
  return n;
}
// The plain generating function G(z) = prod_e sum_{k=0}^{n_e} z^{k m_e}:
// coefficients by the bounded-knapsack difference equation
//   N_k(M) = N_{k-1}(M) + N_k(M - m_k) - N_{k-1}(M - (n_k+1) m_k).
void countAll1D(const Formula& ion, std::vector<double>& N) {
  int Mmax = nomOf(ion);
  N.assign(Mmax + 1, 0.0);
  N[0] = 1;
  std::vector<double> prev;
  for (int e = 0; e < NE; ++e) {
    if (!ion[e]) continue;
    prev = N;
    int m = NOM[e], m2 = (ion[e] + 1) * NOM[e];
    for (int M = 0; M <= Mmax; ++M) {
      double v = prev[M];
      if (M >= m) v += N[M - m];
      if (M >= m2) v -= prev[M - m2];
      N[M] = v;
    }
  }
}
double countAt(const Formula& ion, int M) {
  int maxRestNom[10];
  maxRestNom[9] = ion[C] * 12 + ion[H];
  for (int k = 8; k >= 0; --k) maxRestNom[k] = maxRestNom[k + 1] + ion[HEAVY[k]] * NOM[HEAVY[k]];
  return countRec(ion, M, 0, 0, 0, false, maxRestNom);
}

// Sum over peaks of w_i * log LR_i(F) for one spectrum; -inf if the ion is invalid.
double logLik(const Formula& F, const Spec& sp, const Params& P, SubCounter& cnt,
              std::vector<Match>& buf) {
  Adduct a = adductOf(sp.adduct);
  Formula ion = F;
  for (int e = 0; e < NE; ++e) { ion[e] += a.delta[e]; if (ion[e] < 0) return -INFINITY; }
  if (P.dpcount) cnt.build(ion);
  double L = 0;
  Formula g; g.fill(0);
  for (const Peak& pk : sp.pk) {
    buf.clear();
    double tol = 4 * pk.sig;
    matchAll(ion, a.z, pk.mz + a.z * ELECTRON, tol, buf);
    double s = 0;
    int lastNom = -1;
    double lastN = 0;
    for (const Match& m : buf) {
      double dz = (pk.mz - m.mass) / pk.sig;
      double phi = std::exp(-0.5 * dz * dz) / (pk.sig * 2.5066282746);
      if (m.nom != lastNom) { lastNom = m.nom; lastN = P.dpcount ? cnt.count(m.nom) : countAt(ion, m.nom); }
      double n = lastN;
      if (n < 1) n = 1;
      s += phi / (n * qdens(P, m.nom, pk.mz - m.nom));
    }
    L += pk.w * std::log((1 - P.pi) + P.pi * s);
  }
  return L;
}

struct Data {
  std::vector<Head> head;
  std::vector<std::string> names, smi;
  std::vector<Spec> spec;
  std::map<int, std::vector<int>> specOf;  // mol -> usable spectra
};

Data loadQry(const std::string& dir, const std::string& qs, const Params& P) {
  Data d;
  std::vector<char> hb = slurp(dir + "/" + qs + "_head.bin");
  d.head.resize(hb.size() / sizeof(Head));
  std::memcpy(d.head.data(), hb.data(), hb.size());
  std::vector<char> pb = slurp(dir + "/" + qs + "_peaks.bin");
  d.spec.resize(d.head.size());
  const float* p = reinterpret_cast<const float*>(pb.data());
  for (size_t s = 0; s < d.head.size(); ++s) {
    const Head& h = d.head[s];
    std::vector<std::pair<double, double>> pk;
    for (int i = 0; i < h.n; ++i) pk.emplace_back(p[i], p[h.n + i]);
    p += 2 * h.n;
    pk.erase(std::remove_if(pk.begin(), pk.end(),
                            [&](auto& x) { return x.first > h.precursor - 0.5; }), pk.end());
    std::sort(pk.begin(), pk.end(), [](auto& a, auto& b) { return a.second > b.second; });
    if ((int)pk.size() > P.maxpk) pk.resize(P.maxpk);
    double mx = 0;
    for (auto& x : pk) mx = std::max(mx, std::sqrt(std::max(0.0, x.second)));
    Spec sp;
    sp.adduct = adductIdx(h.tag);
    for (auto& x : pk)
      sp.pk.push_back({x.first, mx > 0 ? std::sqrt(std::max(0.0, x.second)) / mx : 0,
                       std::max(x.first * P.fppm * 1e-6, P.fabs)});
    d.spec[s] = sp;
    if (adductOf(sp.adduct).ok) d.specOf[h.mol].push_back(int(s));
  }
  std::ifstream f(dir + "/" + qs + "_mols.txt");
  std::string line;
  while (std::getline(f, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    size_t t = line.find('\t');
    d.names.push_back(line.substr(0, t));
    d.smi.push_back(t == std::string::npos ? "" : line.substr(t + 1));
  }
  return d;
}

double neutralMass(const Head& h) {
  Adduct a = adductOf(adductIdx(h.tag));
  return h.precursor + a.z * ELECTRON - massOf(a.delta);
}

void parseKV(int argc, char** argv, int from, Params& P) {
  for (int i = from; i < argc; ++i) {
    std::string s = argv[i];
    size_t eq = s.find('=');
    if (eq == std::string::npos) continue;
    std::string k = s.substr(0, eq), v = s.substr(eq + 1);
    if (k == "ppm") P.ppm = std::atof(v.c_str());
    else if (k == "fppm") P.fppm = std::atof(v.c_str());
    else if (k == "fabs") P.fabs = std::atof(v.c_str());
    else if (k == "pi") P.pi = std::atof(v.c_str());
    else if (k == "maxpk") P.maxpk = std::atoi(v.c_str());
    else if (k == "alpha") P.alpha = std::atof(v.c_str());
    else if (k == "qunif") P.qunif = std::atof(v.c_str());
    else if (k == "q") std::sscanf(v.c_str(), "%lf,%lf,%lf,%lf", &P.qa, &P.qb, &P.qc, &P.qd);
    else if (k == "poolf") P.poolf = v;
    else if (k == "qfile") {
      std::ifstream f(v);
      double x, y, z;
      while (f >> x >> y >> z) P.qtab.push_back({x, y, z});
      if (P.qtab.size() < 2) { std::fprintf(stderr, "bad qfile %s\n", v.c_str()); std::exit(1); }
    }
    else if (k == "pool") P.pool = v;
    else if (k == "out") P.out = v;
    else if (k == "outvar") P.outvar = v;
    else if (k == "outT") P.outT = std::atof(v.c_str());
    else if (k == "dpcount") P.dpcount = std::atoi(v.c_str());
    else if (k == "prune") P.prune = std::atof(v.c_str());
    else if (k == "rdbecount") P.rdbecount = std::atoi(v.c_str());
    else if (k == "occam") P.occam = std::atoi(v.c_str());
    else std::fprintf(stderr, "unknown key %s\n", k.c_str());
  }
}

template <class Fn>
void parallelFor(size_t n, Fn fn) {
  std::atomic<size_t> next(0);
  std::vector<std::thread> th;
  for (unsigned t = 0; t < std::max(1u, std::thread::hardware_concurrency()); ++t)
    th.emplace_back([&]() {
      for (;;) {
        size_t i = next++;
        if (i >= n) break;
        fn(i);
      }
    });
  for (auto& t : th) t.join();
}

double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------------------
int cmdPoolCache(const std::string& dir, const std::string& stem, const std::string& outPath) {
  std::ifstream f(dir + "/" + stem + "_mols.txt");
  std::vector<std::string> smis;
  std::string line;
  while (std::getline(f, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    size_t t = line.find('\t');
    smis.push_back(t == std::string::npos ? "" : line.substr(t + 1));
  }
  std::vector<std::string> form(smis.size());
  RDKit::PeriodicTable::getTable();
  double t0 = now();
  parallelFor(smis.size(), [&](size_t i) {
    bool ok;
    Formula fm = formulaOfSmiles(smis[i], ok);
    if (ok) form[i] = hill(fm);
  });
  std::ofstream o(outPath);
  size_t nok = 0;
  for (auto& s : form) { o << (s.empty() ? "-" : s) << "\n"; nok += !s.empty(); }
  std::printf("pool %s: %zu structures, %zu with formulas (%.1fs)\n", stem.c_str(), smis.size(), nok,
              now() - t0);
  return 0;
}

// Fit the defect model q from peaks explained by the true formula.
int cmdFitQ(const std::string& dir, const std::string& qs, const Params& P) {
  RDKit::PeriodicTable::getTable();
  Data d = loadQry(dir, qs, P);
  std::mutex mu;
  std::vector<std::array<double, 3>> pts;  // M, defect, ppm error
  std::vector<int> mols;
  for (auto& kv : d.specOf) mols.push_back(kv.first);
  parallelFor(mols.size(), [&](size_t mi) {
    int mol = mols[mi];
    bool ok;
    Formula tf = formulaOfSmiles(d.smi[mol], ok);
    if (!ok) return;
    std::vector<std::array<double, 3>> loc;
    std::vector<Match> buf;
    Formula g; g.fill(0);
    for (int s : d.specOf[mol]) {
      const Spec& sp = d.spec[s];
      Adduct a = adductOf(sp.adduct);
      Formula ion = tf;
      bool neg = false;
      for (int e = 0; e < NE; ++e) { ion[e] += a.delta[e]; neg |= ion[e] < 0; }
      if (neg) continue;
      for (const Peak& pk : sp.pk) {
        buf.clear();
        matchAll(ion, a.z, pk.mz + a.z * ELECTRON, 4 * pk.sig, buf);
        if (buf.empty()) continue;
        const Match* best = &buf[0];
        for (auto& m : buf) if (std::fabs(m.mass - pk.mz) < std::fabs(best->mass - pk.mz)) best = &m;
        loc.push_back({double(best->nom), pk.mz - best->nom, (pk.mz - best->mass) / pk.mz * 1e6});
      }
    }
    std::lock_guard<std::mutex> lk(mu);
    pts.insert(pts.end(), loc.begin(), loc.end());
  });
  // OLS d = a + b M; then |res| = (c + d M) * sqrt(2/pi).
  auto ols = [&](auto yf, double& a, double& b) {
    double n = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (auto& p : pts) { double x = p[0], y = yf(p); n++; sx += x; sy += y; sxx += x * x; sxy += x * y; }
    b = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    a = (sy - b * sx) / n;
  };
  double a, b, c, dd;
  ols([](auto& p) { return p[1]; }, a, b);
  ols([&](auto& p) { return std::fabs(p[1] - a - b * p[0]) * 1.2533141373; }, c, dd);
  std::vector<double> err;
  for (auto& p : pts) err.push_back(std::fabs(p[2]));
  std::sort(err.begin(), err.end());
  auto q = [&](double f) { return err.empty() ? 0 : err[size_t(f * (err.size() - 1))]; };
  std::printf("matched peaks %zu\n", pts.size());
  std::printf("defect model: mean %.6f + %.7f*M ; sd %.6f + %.7f*M\n", a, b, c, dd);
  std::printf("q=%.6f,%.7f,%.6f,%.7f\n", a, b, c, dd);
  std::printf("|ppm error| quantiles: 50%% %.2f  68%% %.2f  90%% %.2f  95%% %.2f\n", q(0.5), q(0.68), q(0.9),
              q(0.95));
  // Absolute error (mDa) quantiles per mass bin: what sigma should be.
  for (int lo = 0; lo < 1000; lo += 100) {
    std::vector<double> e;
    for (auto& p : pts) if (p[0] >= lo && p[0] < lo + 100) e.push_back(std::fabs(p[2]) * (p[0] + 0.3) * 1e-3);
    std::sort(e.begin(), e.end());
    if (e.size() > 10)
      std::printf("  M %4d-%4d: n %6zu |err| mDa 50%% %.2f 68%% %.2f 90%% %.2f\n", lo, lo + 100, e.size(),
                  e[e.size() / 2], e[size_t(0.68 * e.size())], e[size_t(0.9 * e.size())]);
  }
  // Binned defect table (50 Da), with the linear fit where data is thin.
  std::vector<std::array<double, 3>> tab;
  for (int lo = 0; lo < 2000; lo += 50) {
    double n = 0, s1 = 0, s2 = 0;
    for (auto& p : pts) if (p[0] >= lo && p[0] < lo + 50) { n++; s1 += p[1]; s2 += p[1] * p[1]; }
    double cM = lo + 25, mu = a + b * cM, sd = c + dd * cM;
    if (n >= 40) { mu = s1 / n; sd = std::sqrt(std::max(1e-6, s2 / n - mu * mu)); }
    tab.push_back({cM, mu, sd});
    if (lo < 1200 && n > 1)
      std::printf("  M %4d-%4d: n %6.0f defect mean %.4f sd %.4f (linear %.4f / %.4f)\n", lo, lo + 50, n,
                  mu, sd, a + b * cM, c + dd * cM);
  }
  if (!P.out.empty()) {
    std::ofstream o(P.out);
    for (auto& t : tab) o << t[0] << " " << t[1] << " " << t[2] << "\n";
    std::printf("wrote defect table %s\n", P.out.c_str());
  }
  return 0;
}

struct Cand {
  Formula f;
  std::string name;
  bool enumd = false;
  int poolCnt = 0;
  double ll = 0;  // sum over spectra of w log LR
};

// Score all candidates of one molecule: per candidate and spectrum, the
// 1-D coefficients N_F(M) by the difference equation, and per peak the
// exact-mass sub-formula matches (heavy atoms recursively, C/H solved).
void scoreMolecule(std::vector<Cand>& cands, const std::vector<int>& use, const Data& d, const Params& P,
                   int& nFull) {
  std::vector<Match> buf;
  std::vector<double> G;
  SubCounter cnt;
  nFull = 0;
  for (Cand& cd : cands) {
    double L = 0;
    for (int si : use) {
      const Spec& sp = d.spec[si];
      Adduct a = adductOf(sp.adduct);
      Formula ion = cd.f;
      bool bad = false;
      for (int e = 0; e < NE; ++e) { ion[e] += a.delta[e]; bad |= ion[e] < 0; }
      if (bad) { L = -INFINITY; break; }
      if (P.rdbecount) cnt.build(ion); else countAll1D(ion, G);
      for (const Peak& pk : sp.pk) {
        buf.clear();
        matchAll(ion, a.z, pk.mz + a.z * ELECTRON, 4 * pk.sig, buf);
        double s = 0;
        for (const Match& m : buf) {
          double dz = (pk.mz - m.mass) / pk.sig;
          double phi = std::exp(-0.5 * dz * dz) / (pk.sig * 2.5066282746);
          double n = P.occam ? std::max(1.0, P.rdbecount ? cnt.count(m.nom) : G[m.nom]) : 1.0;
          s += phi / (n * qdens(P, m.nom, pk.mz - m.nom));
        }
        L += pk.w * std::log((1 - P.pi) + P.pi * s);
      }
    }
    cd.ll = L;
    ++nFull;
  }
}

int cmdEval(const std::string& dir, const std::string& qs, const Params& P) {
  RDKit::PeriodicTable::getTable();
  double t0 = now();
  Data d = loadQry(dir, qs, P);

  // Pool: per structure (mass, formula id, key).
  struct PoolEnt { double mass; int fid; int keyId; };
  std::vector<PoolEnt> pool;
  std::vector<std::string> fnames, keys;
  {
    std::ifstream ff(P.poolf);
    std::ifstream fm(dir + "/" + P.pool + "_mols.txt");
    if (!ff || !fm) { std::fprintf(stderr, "need poolf= and pool mols\n"); return 1; }
    std::unordered_map<std::string, int> fidOf;
    std::string lf, lm;
    while (std::getline(ff, lf) && std::getline(fm, lm)) {
      if (!lf.empty() && lf.back() == '\r') lf.pop_back();
      size_t t = lm.find('\t');
      std::string key = lm.substr(0, t);
      if (lf == "-") continue;
      Formula fo;
      if (!parseHill(lf, fo)) continue;
      auto it = fidOf.find(lf);
      int fid;
      if (it == fidOf.end()) { fid = int(fnames.size()); fidOf.emplace(lf, fid); fnames.push_back(lf); }
      else fid = it->second;
      keys.push_back(key);
      pool.push_back({massOf(fo), fid, int(keys.size() - 1)});
    }
    std::sort(pool.begin(), pool.end(), [](auto& a, auto& b) { return a.mass < b.mass; });
    std::printf("pool %zu structures, %zu formulas (%.1fs)\n", pool.size(), fnames.size(), now() - t0);
  }

  std::vector<int> mols;
  for (auto& kv : d.specOf) mols.push_back(kv.first);
  struct Res {
    bool truthOk = false;
    Formula tf{};
    std::vector<Cand> cands;
    std::vector<std::pair<int, int>> poolWin;  // (keyId, fid) in window
    double sec = 0;
    int nFull = 0;
  };
  std::vector<Res> res(mols.size());
  double t1 = now();
  parallelFor(mols.size(), [&](size_t mi) {
    double ts = now();
    int mol = mols[mi];
    Res& r = res[mi];
    bool ok;
    r.tf = formulaOfSmiles(d.smi[mol], ok);
    r.truthOk = ok;
    const std::vector<int>& use = d.specOf[mol];
    double M = neutralMass(d.head[use[0]]);
    std::unordered_map<std::string, int> idx;
    for (const Formula& f : enumerate(M, P.ppm)) {
      std::string nm = hill(f);
      idx.emplace(nm, int(r.cands.size()));
      Cand c; c.f = f; c.name = nm; c.enumd = true;
      r.cands.push_back(c);
    }
    double tol = M * P.ppm * 1e-6;
    auto lo = std::lower_bound(pool.begin(), pool.end(), M - tol,
                               [](const PoolEnt& p, double v) { return p.mass < v; });
    for (auto it = lo; it != pool.end() && it->mass <= M + tol; ++it) {
      r.poolWin.push_back({it->keyId, it->fid});
      const std::string& nm = fnames[it->fid];
      auto f = idx.find(nm);
      if (f == idx.end()) {
        Cand c; parseHill(nm, c.f); c.name = nm; c.poolCnt = 1;
        idx.emplace(nm, int(r.cands.size()));
        r.cands.push_back(c);
      } else {
        r.cands[f->second].poolCnt++;
      }
    }
    scoreMolecule(r.cands, use, d, P, r.nFull);
    r.sec = now() - ts;
    if (r.sec > 5)
      std::fprintf(stderr, "slow: %s M %.2f cands %zu spectra %zu %.1fs\n", d.names[mol].c_str(), M,
                   r.cands.size(), use.size(), r.sec);
  });
  double tEval = now() - t1;

  // Variants: which candidates, and their log prior.
  struct Var { const char* name; int mode; double alpha; };
  // mode 0: enumerated, uniform. 1: pool formulas, log count. 2: pool formulas, uniform.
  // 3: union, log(count + alpha).
  std::vector<Var> vars = {{"A_enum_uniform", 0, 0}, {"B_pool_count", 1, 0}, {"B0_pool_uniform", 2, 0},
                           {"C_union_a0.1", 3, 0.1}, {"C_union_a1", 3, 1.0}, {"U_union_uniform", 4, 0}};
  auto prior = [](const Var& v, const Cand& c) -> double {
    switch (v.mode) {
      case 0: return c.enumd ? 0.0 : -INFINITY;
      case 1: return c.poolCnt > 0 ? std::log(double(c.poolCnt)) : -INFINITY;
      case 2: return c.poolCnt > 0 ? 0.0 : -INFINITY;
      case 4: return 0.0;
      default: return std::log(c.poolCnt + v.alpha);
    }
  };
  std::string trueName;
  const double Ts[] = {0, 0.25, 0.5, 1, 2, 4};
  std::printf("%s: %zu molecules, scoring %.1fs wall\n", qs.c_str(), mols.size(), tEval);
  double nc = 0, sec = 0;
  for (auto& r : res) { sec += r.sec; }
  std::printf("mean CPU sec / molecule %.3f\n", sec / mols.size());
  {
    double nf = 0, na = 0;
    for (auto& r : res) { nf += r.nFull; na += r.cands.size(); }
    std::printf("candidates scored exactly %.1f%% (rest bounded, prune=%g)\n", 100 * nf / std::max(1.0, na),
                P.prune);
  }

  auto logpostOf = [&](const Var& v, double T, const Res& r, std::vector<double>& lp) {
    lp.assign(r.cands.size(), -INFINITY);
    double mx = -INFINITY;
    for (size_t i = 0; i < r.cands.size(); ++i) {
      double pr = prior(v, r.cands[i]);
      if (!std::isfinite(pr) || !std::isfinite(r.cands[i].ll)) continue;
      lp[i] = pr + T * r.cands[i].ll;
      mx = std::max(mx, lp[i]);
    }
    if (!std::isfinite(mx)) return;
    double z = 0;
    for (double x : lp) if (std::isfinite(x)) z += std::exp(x - mx);
    for (double& x : lp) if (std::isfinite(x)) x -= mx + std::log(z);
  };

  for (const Var& v : vars) {
    nc = 0;
    int ncand = 0;
    for (auto& r : res) for (auto& c : r.cands) if (std::isfinite(prior(v, c))) ++ncand;
    std::printf("\n== %s: mean #candidates %.1f\n", v.name, double(ncand) / res.size());
    for (double T : Ts) {
      int n = 0, in = 0, t1c = 0, t3 = 0, t5 = 0, inWin = 0;
      double sumLp = 0, mrr = 0;
      std::vector<double> lp;
      for (size_t mi = 0; mi < res.size(); ++mi) {
        const Res& r = res[mi];
        if (!r.truthOk) continue;
        ++n;
        logpostOf(v, T, r, lp);
        std::string tn = hill(r.tf);
        int ti = -1;
        for (size_t i = 0; i < r.cands.size(); ++i) if (r.cands[i].name == tn) ti = int(i);
        if (ti >= 0 && std::isfinite(lp[ti])) {
          ++in;
          sumLp += std::max(lp[ti], -30.0);
          int better = 0, tie = 0;
          for (size_t i = 0; i < r.cands.size(); ++i) {
            if (!std::isfinite(lp[i])) continue;
            if (lp[i] > lp[ti] + 1e-12) ++better;
            else if (std::fabs(lp[i] - lp[ti]) <= 1e-12) ++tie;
          }
          if (better == 0 && tie == 1) ++t1c;
          if (better + tie <= 3) ++t3;
          if (better + tie <= 5) ++t5;
        }
        // Pool rerank by formula logpost: expected RR@25 with random ties.
        std::unordered_map<std::string, double> sOf;
        for (size_t i = 0; i < r.cands.size(); ++i)
          sOf[r.cands[i].name] = std::isfinite(lp[i]) ? lp[i] : -1e300;
        std::unordered_map<std::string, double> best;  // InChIKey14 -> score
        bool trueKey = false;
        const std::string& qk = d.names[mols[mi]];
        for (auto& pw : r.poolWin) {
          double s = sOf.count(fnames[pw.second]) ? sOf[fnames[pw.second]] : -1e300;
          const std::string& k = keys[pw.first];
          auto it = best.find(k);
          if (it == best.end() || s > it->second) best[k] = s;
          if (k == qk) trueKey = true;
        }
        if (trueKey) {
          ++inWin;
          double st = best[qk];
          int better = 0, tie = 0;
          for (auto& kv : best) {
            if (kv.second > st) ++better;
            else if (kv.second == st) ++tie;
          }
          double rr = 0;
          for (int j = 1; j <= tie; ++j) if (better + j <= 25) rr += 1.0 / (better + j);
          mrr += rr / tie;
        }
      }
      std::printf("  T=%-4g n %d truth-in-cands %.1f%%  top1 %.1f%% top3 %.1f%% top5 %.1f%%  "
                  "mean logp(truth|in) %.3f  poolMRR@25 %.4f (true in window %d)\n",
                  T, n, 100.0 * in / n, 100.0 * t1c / n, 100.0 * t3 / n, 100.0 * t5 / n,
                  sumLp / std::max(1, in), mrr / n, inWin);
    }
  }

  if (!P.out.empty()) {
    const Var* ov = nullptr;
    for (auto& v : vars) if (std::string(v.name).rfind(P.outvar, 0) == 0) { ov = &v; break; }
    if (!ov) ov = &vars[1];
    std::ofstream o(P.out);
    std::vector<double> lp;
    for (size_t mi = 0; mi < res.size(); ++mi) {
      const Res& r = res[mi];
      logpostOf(*ov, P.outT, r, lp);
      std::vector<std::pair<double, int>> v;
      for (size_t i = 0; i < r.cands.size(); ++i) if (std::isfinite(lp[i])) v.push_back({lp[i], int(i)});
      std::sort(v.begin(), v.end(), [](auto& a, auto& b) { return a.first > b.first; });
      o << "Q " << d.names[mols[mi]] << " " << v.size() << "\n";
      char b[64];
      for (auto& x : v) { std::snprintf(b, sizeof b, "%.4f", x.first); o << r.cands[x.second].name << " " << b << "\n"; }
    }
    std::printf("wrote %s (variant %s, T=%g)\n", P.out.c_str(), ov->name, P.outT);
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr,
                 "usage: formula_infer poolcache <dir> <pool_stem> <out>\n"
                 "       formula_infer fitq <dir> <qry> [key=value...]\n"
                 "       formula_infer eval <dir> <qry> poolf=<cache> [key=value...]\n");
    return 1;
  }
  std::string cmd = argv[1];
  Params P;
  parseKV(argc, argv, 4, P);
  if (cmd == "poolcache") return argc > 4 ? cmdPoolCache(argv[2], argv[3], argv[4]) : 1;
  if (cmd == "fitq") return cmdFitQ(argv[2], argv[3], P);
  if (cmd == "selftest") {
    // Direct N(M) must equal the generating-function DP for every M.
    const char* fs[] = {"C20H25N3O5", "C30H48O12Na", "C12H9Cl2NOS", "C25H30BrFN4O3P", "C45H73O20"};
    for (const char* f : fs) {
      Formula ion; parseHill(f, ion);
      SubCounter sc; sc.build(ion);
      int bad = 0; double tot = 0;
      for (int M = 4; M <= nomOf(ion); ++M) { double a = sc.count(M), b = countAt(ion, M); tot += b; if (a != b) ++bad; }
      std::printf("%s: total sub-formulas %.0f, mismatching masses %d\n", f, tot, bad);
    }
    return 0;
  }
  if (cmd == "eval") return cmdEval(argv[2], argv[3], P);
  std::fprintf(stderr, "unknown command\n");
  return 1;
}
