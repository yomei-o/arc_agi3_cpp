// phys_ana: phys_rank + shift-aware (modified-cosine) analog search.
//
// Analog search in two stages per query spectrum:
//   stage 1: two inverted indexes over 0.01 Da bins of all library spectra -
//     fragment m/z (catches fragments that do NOT carry the modification:
//     same m/z in query and analog) and neutral loss precursor - m/z (catches
//     fragments that DO carry it: both fragment and precursor move by Delta,
//     so the loss is invariant). Sum of both scores ~ an upper estimate of the
//     modified cosine; the top K library spectra (|Delta| <= 200) go on.
//   stage 2: exact modified cosine (sqrt intensity, 0.01 Da, greedy
//     one-to-one over unshifted and Delta-shifted pairs) on the shortlist.
// Analog feature variants (max_a Tanimoto * sim_a^p, top 80 molecules):
//   0 old binned unshifted (phys_rank), 1 exact unshifted cosine,
//   2 modified cosine, 3 modified cosine without precursor-region peaks,
//   4 modified cosine requiring >= 6 matched peaks, 5-7 variant 3 requiring
//   >= 4/6/8 matched peaks, 8 exact cosine requiring >= 6 matched peaks.
//   extra args after [features.txt]: [anaVariant] [anaPow] [K] [weights.txt]
//   recommended: wa 16-24, variant 4, pow 3, K 8000 (K saturates by 8000;
//   cqry 0.484 -> 0.550, tqry 0.534 -> 0.612, sqry 0.839 -> 0.840)
//
// Physics-shaped fragment intensities instead of a learned MLP.
//
// The learned MLP adds +0.19 MRR over equal-intensity fragments on the
// synthetic holdout (dqry) but only +0.047 on natural products (cqry): what
// it learned is mostly class-specific. This replaces it with ~20 parameters
// shaped by gas-phase ion chemistry (PHYSCHEM_IDEAS.md), fitted on one
// natural-product holdout and evaluated on another:
//
//   intensity(fragment X of cut c, H-shift k)
//       = exp(-E[class(c)]) * share(X | c) * w_k
//   share(X | c) = exp(beta * S(X)) / sum_{Y in c} exp(beta * S(Y))
//
//   - Cleavage propensity: one barrier E per bond class (acyl C-O/C-N,
//     glycosidic, other C-heteroatom, aryl/benzylic C-C, C-C, ring opening,
//     ring opening through an aromatic bond, other), an Arrhenius factor at
//     one effective temperature folded into E.
//   - Charge retention (Cooks' kinetic method): the fragment with the higher
//     proton affinity keeps the proton, ln(I_X/I_Y) = beta (PA_X - PA_Y);
//     in negative mode the more acidic fragment keeps the charge. S(X) is a
//     soft maximum over the fragment's sites from a small functional-group
//     table (kcal/mol).
//   - Hydrogen rearrangement: a weight per shift k in -2..2, per polarity.
//
// Scoring is the background-mixture log-likelihood of pool_candidate_rank
// (loglik), mean over the molecule's spectra.
//
//   phys_rank fit  <dir> <qry> <pool_stem> <ppm> <params_out> [params_init] [instrument]
//   phys_rank eval <dir> <qry> <pool_stem> <ppm> <params_in> [out.csv] [mlp.bin] [w] [smiles]
//                  [rev.bin] [wr] [lib_stem] [wl] [lib_pow] [wa] [features.txt]
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <GraphMol/GraphMol.h>
#include <GraphMol/PeriodicTable.h>
#include <GraphMol/RingInfo.h>
#include <GraphMol/SmilesParse/SmilesParse.h>

#include "forward_model.h"
#include "forward_model_nn.h"
#include "rev/ecfp.h"  // copies of ../rev headers (the dataset ships them inside rdkit-min/)
#include "rev/rev_model.hpp"

namespace {

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

constexpr float ADDUCT_SHIFT[10] = {
    1.007276f, -1.007276f, 22.989218f, 18.033823f, 38.963158f,
    34.969402f, 44.998201f, -17.002740f, 1.007276f, 22.989218f,
};
constexpr bool ADDUCT_NEG[10] = {false, true, false, false, false, true, true, false, false, false};
constexpr bool ADDUCT_METAL[10] = {false, false, true, false, true, false, false, false, false, true};

int adductIdx(int32_t tag) {
  if (tag < 0) return -1;
  return tag % 16 == 15 ? tag / 16 + 1 : tag / 16;
}

// ---- site tables -------------------------------------------------------

bool hasDoubleO(const RDKit::ROMol& m, const RDKit::Atom* c) {
  for (auto b : m.atomBonds(c)) {
    const RDKit::Atom* o = b->getOtherAtom(c);
    if (o->getAtomicNum() == 8 && b->getBondType() == RDKit::Bond::DOUBLE) return true;
  }
  return false;
}

// Gas-phase proton affinity of a protonation site, kcal/mol (0 = not a site).
float sitePA(const RDKit::ROMol& m, const RDKit::Atom* a) {
  int z = a->getAtomicNum();
  if (z == 7) {
    if (a->getIsAromatic()) return a->getTotalNumHs() > 0 ? 190.f : 222.f;  // pyrrole / pyridine
    bool amide = false, aryl = false, imine = false;
    int nC = 0;
    for (auto b : m.atomBonds(a)) {
      const RDKit::Atom* o = b->getOtherAtom(a);
      if (b->getBondType() == RDKit::Bond::DOUBLE) imine = true;
      if (o->getAtomicNum() == 6) {
        ++nC;
        if (hasDoubleO(m, o)) amide = true;
        if (o->getIsAromatic()) aryl = true;
      }
    }
    if (amide) return 200.f;   // amides protonate on O (below)
    if (imine) return 225.f;
    if (aryl) return 210.f;
    return 214.f + 4.f * float(std::min(nC, 3));  // primary < secondary < tertiary
  }
  if (z == 8) {
    bool dbl = false;
    for (auto b : m.atomBonds(a))
      if (b->getBondType() == RDKit::Bond::DOUBLE) {
        dbl = true;
        const RDKit::Atom* c = b->getOtherAtom(a);
        bool nNb = false, oNb = false, arNb = false;
        for (auto b2 : m.atomBonds(c)) {
          const RDKit::Atom* x = b2->getOtherAtom(c);
          if (x == a) continue;
          if (x->getAtomicNum() == 7) nNb = true;
          if (x->getAtomicNum() == 8) oNb = true;
          if (x->getIsAromatic()) arNb = true;
        }
        if (nNb) return 212.f;              // amide carbonyl
        if (oNb) return 196.f;              // ester / acid carbonyl
        return arNb ? 205.f : 197.f;        // conjugated / plain ketone
      }
    if (!dbl) {
      bool ar = false;
      for (auto b : m.atomBonds(a)) if (b->getOtherAtom(a)->getIsAromatic()) ar = true;
      if (ar) return 195.f;                  // phenol / aryl ether
      return a->getTotalNumHs() > 0 ? 189.f : 192.f;  // alcohol / ether
    }
  }
  if (z == 16) return 200.f;
  if (z == 6 && a->getIsAromatic()) return 185.f;
  return 0.f;
}

// Deprotonation favourability, 400 - gas-phase acidity (kcal/mol); larger =
// more acidic (0 = not a site).
float siteAcid(const RDKit::ROMol& m, const RDKit::Atom* a) {
  int z = a->getAtomicNum();
  if (a->getTotalNumHs() == 0) return 0.f;
  if (z == 8) {
    for (auto b : m.atomBonds(a)) {
      const RDKit::Atom* c = b->getOtherAtom(a);
      if (c->getAtomicNum() == 6 && hasDoubleO(m, c)) return 55.f;      // COOH
      if (c->getAtomicNum() == 16 || c->getAtomicNum() == 15) return 70.f;  // S/P acids
      if (c->getIsAromatic()) return 50.f;                              // phenol
    }
    return 28.f;                                                        // alcohol
  }
  if (z == 7) {
    for (auto b : m.atomBonds(a)) {
      const RDKit::Atom* c = b->getOtherAtom(a);
      if ((c->getAtomicNum() == 6 && hasDoubleO(m, c)) || c->getAtomicNum() == 16) return 40.f;  // amide/sulfonamide NH
    }
    return a->getIsAromatic() ? 40.f : 10.f;
  }
  if (z == 16) return 45.f;                                             // thiol
  return 0.f;
}

// ---- bond classes ------------------------------------------------------

// CGLYC: the sugar-aglycone C-C bond of a C-glycoside (split out of ARYLC/CC
// to test whether the fit recovers "C-glycosides are harder to cleave than
// O-glycosides" by itself).
enum Cls { INTACT = 0, ACYL, GLYCOSIDIC, CHET, ARYLC, CC, RING, RINGAR, OTHER, CGLYC, NCLS };

bool anomeric(const RDKit::ROMol& m, const RDKit::Atom* c) {
  if (c->getAtomicNum() != 6 || c->getIsAromatic()) return false;
  int nO = 0;
  for (auto b : m.atomBonds(c))
    if (b->getOtherAtom(c)->getAtomicNum() == 8 && b->getBondType() == RDKit::Bond::SINGLE) ++nO;
  return nO >= 2 && m.getRingInfo()->numAtomRings(c->getIdx()) > 0;
}

// sp3 ring carbon carrying exactly one (ring) oxygen: a sugar's C1 when the
// aglycone is attached through carbon
bool cAnomeric(const RDKit::ROMol& m, const RDKit::Atom* c) {
  if (c->getAtomicNum() != 6 || c->getIsAromatic()) return false;
  const RDKit::RingInfo* ri = m.getRingInfo();
  if (ri->numAtomRings(c->getIdx()) == 0) return false;
  int nO = 0, nRingO = 0;
  for (auto b : m.atomBonds(c)) {
    const RDKit::Atom* o = b->getOtherAtom(c);
    if (o->getAtomicNum() == 8 && b->getBondType() == RDKit::Bond::SINGLE) {
      ++nO;
      if (ri->numBondRings(b->getIdx()) > 0) ++nRingO;
    }
  }
  return nO == 1 && nRingO == 1;
}

int bondClass(const RDKit::ROMol& m, const RDKit::Bond* b) {
  const RDKit::Atom* x = b->getBeginAtom();
  const RDKit::Atom* y = b->getEndAtom();
  int zx = x->getAtomicNum(), zy = y->getAtomicNum();
  if (b->getBondType() != RDKit::Bond::SINGLE) return OTHER;
  if (zx == 6 && zy == 6) {
    if ((cAnomeric(m, x) && y->getIsAromatic()) || (cAnomeric(m, y) && x->getIsAromatic())) return CGLYC;
    return (x->getIsAromatic() || y->getIsAromatic()) ? ARYLC : CC;
  }
  const RDKit::Atom* c = zx == 6 ? x : (zy == 6 ? y : nullptr);
  const RDKit::Atom* h = c == x ? y : x;
  if (!c) return OTHER;
  int zh = h->getAtomicNum();
  if ((zh == 8 || zh == 7) && hasDoubleO(m, c)) return ACYL;
  if (zh == 8 && anomeric(m, c)) return GLYCOSIDIC;
  if (zh == 8 || zh == 7 || zh == 16) return CHET;
  return OTHER;
}

// ---- per-candidate fragment table --------------------------------------

struct Frag {
  float mass;   // neutral sub-graph mass with its own hydrogens
  uint8_t cls;
  float pa, paOther;      // soft-max site PA of this fragment / its partner
  float acid, acidOther;  // same for acidity
  uint8_t cls2 = 255;     // second cut's class for a two-bond (sequential) cleavage
};

float softmax(const std::vector<float>& v, float t = 3.0f) {
  // log-sum-exp at "temperature" t kcal/mol: a fragment with two good sites
  // is slightly more basic than one with a single site
  float mx = 0.f;
  for (float x : v) mx = std::max(mx, x);
  if (mx <= 0.f) return 0.f;
  double s = 0;
  for (float x : v) if (x > 0.f) s += std::exp((x - mx) / t);
  return mx + t * float(std::log(s));
}

std::vector<Frag> enumerate(const RDKit::ROMol& mol) {
  std::vector<Frag> out;
  const int n = int(mol.getNumAtoms());
  std::vector<float> pa(n), ac(n);
  for (int i = 0; i < n; ++i) {
    pa[i] = sitePA(mol, mol.getAtomWithIdx(i));
    ac[i] = siteAcid(mol, mol.getAtomWithIdx(i));
  }
  auto score = [&](const std::vector<int>& idx, const std::vector<float>& s) {
    std::vector<float> v;
    for (int i : idx) if (s[i] > 0.f) v.push_back(s[i]);
    return softmax(v);
  };
  std::vector<int> all(n);
  for (int i = 0; i < n; ++i) all[i] = i;
  out.push_back({float(forward_model::fragmentMass(mol, all)), INTACT, 0.f, 0.f, 0.f, 0.f});

  auto addCut = [&](const std::vector<unsigned>& bonds, int cls, int cls2 = 255) {
    auto frags = forward_model::fragmentsAfterCutting(mol, bonds);
    if (frags.size() < 2) return;
    std::vector<float> p(frags.size()), a(frags.size());
    for (size_t i = 0; i < frags.size(); ++i) { p[i] = score(frags[i], pa); a[i] = score(frags[i], ac); }
    for (size_t i = 0; i < frags.size(); ++i) {
      // partner = best competing fragment of the same cut
      float po = 0.f, ao = 0.f;
      for (size_t j = 0; j < frags.size(); ++j)
        if (j != i) { po = std::max(po, p[j]); ao = std::max(ao, a[j]); }
      out.push_back({float(forward_model::fragmentMass(mol, frags[i])), uint8_t(cls), p[i], po, a[i], ao,
                     uint8_t(cls2)});
    }
  };
  const RDKit::RingInfo* ri = mol.getRingInfo();
  std::vector<std::pair<unsigned, int>> labile;
  for (const auto& b : mol.bonds()) {
    if (ri->numBondRings(b->getIdx()) > 0) continue;
    int c = bondClass(mol, b);
    addCut({unsigned(b->getIdx())}, c);
    if (c == ACYL || c == GLYCOSIDIC || c == CHET) labile.emplace_back(b->getIdx(), c);
  }
  // Two labile bonds at once: the sequential losses natural products are
  // full of (both sugars of a bis-glycoside, sugar + acyl group, ...), which
  // a single cut can never produce. Capped to keep the cost bounded.
  if (labile.size() > 12) labile.resize(12);
  for (size_t i = 0; i < labile.size(); ++i)
    for (size_t j = i + 1; j < labile.size(); ++j)
      addCut({labile[i].first, labile[j].first}, labile[i].second, labile[j].second);
  for (unsigned r = 0; r < ri->numRings(); ++r) {
    const auto& rb = ri->bondRings()[r];
    for (size_t i = 0; i < rb.size(); ++i)
      for (size_t j = i + 1; j < rb.size(); ++j) {
        bool ar = mol.getBondWithIdx(rb[i])->getIsAromatic() || mol.getBondWithIdx(rb[j])->getIsAromatic();
        addCut({unsigned(rb[i]), unsigned(rb[j])}, ar ? RINGAR : RING);
      }
  }
  return out;
}

// ---- parameters ----------------------------------------------------------

struct Params {
  float E[NCLS] = {0.f, 0.5f, 0.3f, 1.0f, 1.5f, 2.0f, 2.0f, 4.0f, 3.0f, 1.5f};  // E[INTACT] = -log(intact weight)
  float betaPos = 0.15f, betaNeg = 0.10f;  // 1/kcal
  float kPos[5] = {-3.f, -1.f, -0.5f, 0.f, -2.f};  // log weight of H shift -2..2
  float kNeg[5] = {-1.f, 0.f, -0.5f, -2.f, -3.f};
  float logEps = std::log(0.3f);
  float E2 = 1.0f;  // extra barrier for the second cut of a two-bond cleavage
  // size effect: every cut's barrier + dofSlope * ln(heavy atoms / 30). The
  // degrees-of-freedom argument predicts dofSlope > 0 - at one collision
  // energy a bigger ion is internally colder and fragments less.
  float dofSlope = 0.0f;

  // file order keeps older files loadable: the original classes, betas,
  // k weights, eps, E2, then E[CGLYC] and dofSlope (added later)
  static constexpr int N = (NCLS - 1) + 2 + 5 + 5 + 1 + 1 + 1 + 1;
  float* at(int i) {
    if (i < NCLS - 1) return &E[i];
    i -= NCLS - 1;
    if (i == 0) return &betaPos;
    if (i == 1) return &betaNeg;
    i -= 2;
    if (i < 5) return &kPos[i];
    i -= 5;
    if (i < 5) return &kNeg[i];
    i -= 5;
    if (i == 0) return &logEps;
    if (i == 1) return &E2;
    if (i == 2) return &E[CGLYC];
    return &dofSlope;
  }
  void save(const std::string& p) {
    std::ofstream f(p);
    for (int i = 0; i < N; ++i) f << *at(i) << "\n";
  }
  bool load(const std::string& p) {
    std::ifstream f(p);
    // files written before E2 existed hold N-1 values; E2 keeps its default
    for (int i = 0; i < N; ++i) if (!(f >> *at(i))) return i >= N - 3;
    return true;
  }
};

constexpr double MASS_H = 1.00782503207, ELECTRON = 0.00054858;

// Predicted (m/z, intensity) for one polarity / adduct.
void predict(const std::vector<Frag>& fr, const Params& P, int adduct, int nHeavy,
             std::vector<std::pair<double, double>>& out) {
  const double dof = P.dofSlope * std::log(std::max(1, nHeavy) / 30.0);
  out.clear();
  bool neg = adduct >= 0 && adduct < 10 && ADDUCT_NEG[adduct];
  bool metal = adduct >= 0 && adduct < 10 && ADDUCT_METAL[adduct];
  const float* kw = neg ? P.kNeg : P.kPos;
  double z = neg ? -1.0 : 1.0;
  for (const Frag& f : fr) {
    double share = 1.0;
    if (f.cls != INTACT) {
      double d = neg ? P.betaNeg * (f.acid - f.acidOther) : P.betaPos * (f.pa - f.paOther);
      if (metal) d = 0.0;  // a metal cation does not follow proton affinity
      share = 1.0 / (1.0 + std::exp(-d));
    }
    double e = P.E[f.cls] + (f.cls2 != 255 ? P.E[f.cls2] + P.E2 : 0.f) + (f.cls != INTACT ? dof : 0.0);
    double base = std::exp(-e) * share;
    for (int k = -2; k <= 2; ++k)
      out.emplace_back(f.mass + k * MASS_H - z * ELECTRON, base * std::exp(kw[k + 2]));
    if (metal) {
      double mm = adduct == 4 || adduct == 9 ? (adduct == 4 ? 38.963158 : 22.989218) : 22.989218;
      out.emplace_back(f.mass + mm, base * 0.5);
    }
  }
  std::sort(out.begin(), out.end());
}

float loglik(const std::vector<std::pair<double, double>>& pred, const float* mz, const float* it,
             int n, float tol, float eps, float range) {
  double tot = 0.0, wsum = 0.0;
  for (auto& pk : pred) tot += pk.second;
  for (int i = 0; i < n; ++i) wsum += std::sqrt(std::max(0.0, double(it[i])));
  if (tot <= 0.0 || wsum <= 0.0) return -1e9f;
  double bg = eps * 2.0 * tol / std::max(1.0f, range);
  double ll = 0.0;
  for (int i = 0; i < n; ++i) {
    auto lo = std::lower_bound(pred.begin(), pred.end(), std::make_pair(double(mz[i]) - tol, -1e300));
    double m = 0.0;
    for (auto p = lo; p != pred.end() && p->first <= mz[i] + tol; ++p) m += p->second;
    double w = std::sqrt(std::max(0.0, double(it[i]))) / wsum;
    ll += w * std::log((1.0 - eps) * m / tot + bg);
  }
  return float(ll);
}

// ---- data ----------------------------------------------------------------

struct Query {
  std::string name;
  std::vector<int> spectra;
  std::vector<int> cand;     // pool indices
  std::vector<std::vector<Frag>> frags;
  std::vector<int> candN;     // heavy atoms per candidate
  int truth = -1;            // position of the true structure in cand, -1 if absent
  std::vector<float> mlpCos; // learned forward MLP's best cosine, when a model is given
  std::vector<float> revLL;  // reverse model: sum of logit(p_bit) over the candidate's ECFP bits
  std::vector<float> logit;  // reverse model's per-bit logits for this query (mean p over spectra)
  std::vector<float> libSim; // best spectral similarity to the candidate's own library spectra (0 if none)
  std::vector<float> anaP;   // analog propagation: max_a Tanimoto(cand, a) * sim_a^3 over spectral analogs a
  std::vector<float> anaV[9][3];  // variant x pow {2,3,4}
};

constexpr int NVAR = 9;
const char* VARNAME[NVAR] = {"binned", "cos", "mcos", "mcos-noprec", "mcos-6pk",
                            "noprec-4pk", "noprec-6pk", "noprec-8pk", "cos-6pk"};

// Exact modified cosine between two spectra (peaks sorted by m/z).
// Weights are sqrt(I)/||sqrt(I)||. Pairs: |q - l| <= tol (unshifted) or
// |q - delta - l| <= tol (shifted, delta = precQ - precL); greedy one-to-one
// by descending product. Fills cos (unshifted only), mcos, mcosNP (precursor
// region peaks m/z > prec - 1 removed from both, renormalized), nMatch.
struct McosOut { float cos, mcos, mcosNP; int nMatch, nMatchU, nMatchNP; };
struct PairT { float s; int i, j; uint8_t shifted, prec; };
void modcos(const float* qm, const float* qw, int nq, float qprec, float qnNP,
            const float* lm, const float* li, int nl, float lprec, float lnrm,
            float tol, std::vector<PairT>& pairs, std::vector<uint8_t>& usedQ, std::vector<uint8_t>& usedL,
            McosOut& o) {
  float delta = qprec - lprec;
  pairs.clear();
  // lib weights on the fly
  auto lw = [&](int j) { return std::sqrt(std::max(0.f, li[j])) / lnrm; };
  double lnNP2 = 0;  // squared norm of non-precursor lib peaks (in normalized units)
  for (int j = 0; j < nl; ++j) if (lm[j] <= lprec - 1.f) { float w = lw(j); lnNP2 += double(w) * w; }
  for (int pass = 0; pass < 2; ++pass) {
    float sh = pass ? delta : 0.f;
    if (pass && std::fabs(delta) <= tol) break;  // identical to unshifted
    int j0 = 0;
    for (int i = 0; i < nq; ++i) {
      float m = qm[i] - sh;
      while (j0 < nl && lm[j0] < m - tol) ++j0;
      for (int j = j0; j < nl && lm[j] <= m + tol; ++j) {
        uint8_t pr = (qm[i] > qprec - 1.f) || (lm[j] > lprec - 1.f);
        pairs.push_back({qw[i] * lw(j), i, j, uint8_t(pass), pr});
      }
    }
  }
  std::sort(pairs.begin(), pairs.end(), [](const PairT& a, const PairT& b) { return a.s > b.s; });
  usedQ.assign(nq, 0); usedL.assign(nl, 0);
  // three greedy passes: all pairs, unshifted only, non-precursor only
  double sm = 0; int nm = 0;
  for (auto& p : pairs) if (!usedQ[p.i] && !usedL[p.j]) { usedQ[p.i] = usedL[p.j] = 1; sm += p.s; ++nm; }
  o.mcos = float(std::min(1.0, sm)); o.nMatch = nm;
  usedQ.assign(nq, 0); usedL.assign(nl, 0);
  double su = 0; int nu = 0;
  for (auto& p : pairs) if (!p.shifted && !usedQ[p.i] && !usedL[p.j]) { usedQ[p.i] = usedL[p.j] = 1; su += p.s; ++nu; }
  o.cos = float(std::min(1.0, su)); o.nMatchU = nu;
  usedQ.assign(nq, 0); usedL.assign(nl, 0);
  double sn = 0; int nn = 0;
  for (auto& p : pairs) if (!p.prec && !usedQ[p.i] && !usedL[p.j]) { usedQ[p.i] = usedL[p.j] = 1; sn += p.s; ++nn; }
  o.nMatchNP = nn;
  double den = double(qnNP) * std::sqrt(lnNP2);
  o.mcosNP = den > 1e-9 ? float(std::min(1.0, sn / den)) : 0.f;
}

// Same cosine as pool_candidate_rank (sqrt-intensity, 0.01 Da).
float mlpCosine(const std::vector<std::pair<double, double>>& pred, const float* mz, const float* it, int n) {
  double dot = 0.0, na = 0.0, nb = 0.0;
  for (auto& pk : pred) na += pk.second;
  for (int i = 0; i < n; ++i) nb += double(it[i]);
  if (na <= 0.0 || nb <= 0.0) return 0.0f;
  for (auto& pk : pred) {
    double best = 0.0;
    for (int i = 0; i < n; ++i)
      if (std::fabs(double(mz[i]) - pk.first) <= 0.01) best = std::max(best, double(it[i]));
    if (best > 0.0) dot += std::sqrt(pk.second) * std::sqrt(best);
  }
  return float(dot / std::sqrt(na * nb + 1e-12));
}

std::vector<float> zscore(const std::vector<float>& v) {
  double m = 0, s = 0;
  for (float x : v) m += x;
  m /= double(std::max<size_t>(1, v.size()));
  for (float x : v) s += (x - m) * (x - m);
  s = std::sqrt(s / double(std::max<size_t>(1, v.size()))) + 1e-6;
  std::vector<float> z(v.size());
  for (size_t i = 0; i < v.size(); ++i) z[i] = float((v[i] - m) / s);
  return z;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 7) {
    std::fprintf(stderr, "usage: phys_rank fit|eval <dir> <qry> <pool_stem> <ppm> <params> [out.csv]\n");
    return 1;
  }
  std::string mode = argv[1], dir = argv[2], qs = argv[3], poolStem = argv[4];
  float ppm = float(std::atof(argv[5]));
  std::string paramPath = argv[6];
  std::string outPath = argc > 7 ? argv[7] : "";
  // eval only: a forward-MLP model to fuse with, z(phys) + w z(mlp cosine)
  std::string mlpPath = argc > 8 ? argv[8] : "";
  float fuseW = argc > 9 ? float(std::atof(argv[9])) : 0.0f;
  bool writeSmiles = argc > 10 && std::string(argv[10]) == "smiles";  // submission format
  bool haveMlp = !mlpPath.empty();
  // eval only: a reverse (spectrum -> ECFP) model, fused as + wr z(rev)
  std::string revPath = argc > 11 ? argv[11] : "";
  float fuseR = argc > 12 ? float(std::atof(argv[12])) : 0.0f;
  bool haveRev = !revPath.empty();
  // eval only: a spectral library (stem); a candidate with reference spectra
  // there gets its best similarity to the query as a continuous score,
  // + wl * libSim (instead of a hard "library top-1 first" rule)
  std::string libStem = argc > 13 ? argv[13] : "";
  float fuseL = argc > 14 ? float(std::atof(argv[14])) : 0.0f;
  float fuseLPow = argc > 15 ? float(std::atof(argv[15])) : 8.0f;
  float libPow = fuseLPow;
  float fuseA = argc > 16 ? float(std::atof(argv[16])) : 0.0f;  // analog-propagation weight
  float anaW = fuseA;
  // [anaVariant] [anaPow] [K]: which analog feature the output uses
  // (default = phys_rank's binned sim^3), and the stage-1 shortlist size
  int anaVar = argc > 18 ? std::atoi(argv[18]) : 0;
  int anaPowI = argc > 19 ? std::atoi(argv[19]) - 2 : 1;
  int shortK = argc > 20 ? std::atoi(argv[20]) : 2000;
  // [weights.txt]: 8 learned fusion weights (fuse_fit) replacing w/wr/wl/wa
  std::string weightsPath = argc > 21 ? argv[21] : "";
  if (anaVar < 0 || anaVar >= NVAR) anaVar = 0;
  if (anaPowI < 0 || anaPowI > 2) anaPowI = 1;
  bool haveLib = !libStem.empty();
  forward_model_nn::MLP mlp;
  if (haveMlp) mlp = forward_model_nn::MLP::load(dir + "/" + mlpPath);

  // spectra
  std::vector<Head> head;
  std::vector<float> mz, it;
  {
    std::vector<char> hb = slurp(dir + "/" + qs + "_head.bin");
    head.resize(hb.size() / sizeof(Head));
    std::memcpy(head.data(), hb.data(), hb.size());
    std::vector<char> pb = slurp(dir + "/" + qs + "_peaks.bin");
    const float* p = reinterpret_cast<const float*>(pb.data());
    for (Head& h : head) {
      h.off = int32_t(mz.size());
      mz.insert(mz.end(), p, p + h.n); p += h.n;
      it.insert(it.end(), p, p + h.n); p += h.n;
    }
  }
  std::vector<std::string> qnames;
  {
    std::ifstream f(dir + "/" + qs + "_mols.txt");
    std::string line;
    while (std::getline(f, line)) {
      if (!line.empty() && line.back() == '\r') line.pop_back();
      qnames.push_back(line.substr(0, line.find('\t')));
    }
  }
  // pool
  std::vector<float> pmass;
  std::vector<int32_t> pid;
  std::vector<std::string> pname, psmi;
  std::vector<uint8_t> pHasSpec;  // by pool index: a train structure (has reference spectra)
  {
    std::vector<char> hb = slurp(dir + "/" + poolStem + "_head.bin");
    size_t n = hb.size() / 12;
    pmass.resize(n); pid.resize(n); pHasSpec.assign(n, 0);
    for (size_t i = 0; i < n; ++i) {
      int32_t hs = 0;
      std::memcpy(&pmass[i], hb.data() + 12 * i, 4);
      std::memcpy(&pid[i], hb.data() + 12 * i + 4, 4);
      std::memcpy(&hs, hb.data() + 12 * i + 8, 4);
      if (pid[i] >= 0 && size_t(pid[i]) < n) pHasSpec[pid[i]] = uint8_t(hs != 0);
    }
    std::ifstream f(dir + "/" + poolStem + "_mols.txt");
    std::string line;
    while (std::getline(f, line)) {
      if (!line.empty() && line.back() == '\r') line.pop_back();
      size_t t = line.find('\t');
      pname.push_back(line.substr(0, t));
      psmi.push_back(t == std::string::npos ? "" : line.substr(t + 1));
    }
  }
  std::unordered_map<std::string, int> poolIdx;
  for (size_t i = 0; i < pname.size(); ++i) poolIdx.emplace(pname[i], int(i));

  std::unordered_map<int, std::vector<int>> specOf;
  // fit only: [instrument] keeps just that instrument's spectra (0 timsTOF,
  // 1 Orbitrap, ...), to compare per-instrument fits
  int instrFilter = (mode == "fit" && argc > 8) ? std::atoi(argv[8]) : -1;
  for (size_t s = 0; s < head.size(); ++s) {
    if (instrFilter >= 0) {
      int32_t t = head[s].tag;
      int ins = (t < 0 || t % 16 == 15) ? -1 : t % 16;
      if (ins != instrFilter) continue;
    }
    specOf[head[s].mol].push_back(int(s));
  }
  if (instrFilter >= 0) std::printf("instrument filter %d: %zu molecules\n", instrFilter, specOf.size());
  std::vector<Query> Q;
  for (auto& kv : specOf) {
    Query q;
    q.name = qnames[kv.first];
    q.spectra = kv.second;
    Q.push_back(std::move(q));
  }
  std::sort(Q.begin(), Q.end(), [](auto& a, auto& b) { return a.name < b.name; });

  // Spectral library: spectra grouped by molecule, keyed by InChIKey14 name.
  constexpr int NBIN = 300000;  // 0.01 Da bins up to 3000 m/z
  std::vector<int64_t> binStart;
  std::vector<int32_t> postSpec;
  std::vector<float> postW;
  std::vector<int64_t> nlStart;
  std::vector<int32_t> nlSpec;
  std::vector<float> nlW, lnrm;
  std::vector<std::string> lnames, lsmiles;
  std::vector<Head> lhead;
  std::vector<float> lmz, lit;
  std::unordered_map<std::string, std::vector<int>> libSpecOf;
  if (haveLib) {
    std::vector<char> hb = slurp(dir + "/" + libStem + "_head.bin");
    lhead.resize(hb.size() / sizeof(Head));
    std::memcpy(lhead.data(), hb.data(), hb.size());
    std::vector<char> pb = slurp(dir + "/" + libStem + "_peaks.bin");
    const float* p = reinterpret_cast<const float*>(pb.data());
    size_t total = 0;
    for (const Head& h : lhead) total += size_t(h.n);
    lmz.reserve(total); lit.reserve(total);
    for (Head& h : lhead) {
      h.off = int32_t(lmz.size());
      lmz.insert(lmz.end(), p, p + h.n); p += h.n;
      lit.insert(lit.end(), p, p + h.n); p += h.n;
    }
    std::ifstream f(dir + "/" + libStem + "_mols.txt");
    std::string line;
    while (std::getline(f, line)) {
      if (!line.empty() && line.back() == '\r') line.pop_back();
      size_t t = line.find('\t');
      lnames.push_back(line.substr(0, t));
      lsmiles.push_back(t == std::string::npos ? "" : line.substr(t + 1));
    }
    for (size_t s = 0; s < lhead.size(); ++s) libSpecOf[lnames[lhead[s].mol]].push_back(int(s));
    std::printf("library %s: %zu spectra, %zu molecules\n", libStem.c_str(), lhead.size(), libSpecOf.size());

    // Inverted index over 0.01 Da bins for the analog search: posting =
    // (spectrum, sqrt(I)/||sqrt(I)||), so summing query x library weights
    // over shared bins approximates the sqrt-intensity cosine of every
    // library spectrum at once.
    binStart.assign(NBIN + 1, 0);
    for (const Head& h : lhead)
      for (int j = 0; j < h.n; ++j) {
        int b = int(lmz[h.off + j] * 100.f);
        if (b >= 0 && b < NBIN) ++binStart[b + 1];
      }
    for (int b = 0; b < NBIN; ++b) binStart[b + 1] += binStart[b];
    postSpec.resize(binStart[NBIN]); postW.resize(binStart[NBIN]);
    std::vector<int64_t> fill(binStart.begin(), binStart.end() - 1);
    for (size_t s = 0; s < lhead.size(); ++s) {
      const Head& h = lhead[s];
      double nrm = 0;
      for (int j = 0; j < h.n; ++j) nrm += std::max(0.f, lit[h.off + j]);
      nrm = std::sqrt(nrm) + 1e-12;
      for (int j = 0; j < h.n; ++j) {
        int b = int(lmz[h.off + j] * 100.f);
        if (b < 0 || b >= NBIN) continue;
        postSpec[fill[b]] = int32_t(s);
        postW[fill[b]++] = float(std::sqrt(std::max(0.f, lit[h.off + j])) / nrm);
      }
    }
    std::printf("analog index: %zu postings\n", postSpec.size());
    // Neutral-loss index: bin of precursor - m/z (fragments below the
    // precursor only, including the precursor peak itself at loss ~0)
    lnrm.resize(lhead.size());
    nlStart.assign(NBIN + 1, 0);
    for (const Head& h : lhead)
      for (int j = 0; j < h.n; ++j) {
        int b = int((h.precursor - lmz[h.off + j]) * 100.f + 0.5f);  // 0.5 so the precursor peak lands in bin 0
        if (b >= 0 && b < NBIN) ++nlStart[b + 1];
      }
    for (int b = 0; b < NBIN; ++b) nlStart[b + 1] += nlStart[b];
    nlSpec.resize(nlStart[NBIN]); nlW.resize(nlStart[NBIN]);
    std::vector<int64_t> fill2(nlStart.begin(), nlStart.end() - 1);
    for (size_t s = 0; s < lhead.size(); ++s) {
      const Head& h = lhead[s];
      double nrm = 0;
      for (int j = 0; j < h.n; ++j) nrm += std::max(0.f, lit[h.off + j]);
      nrm = std::sqrt(nrm) + 1e-12;
      lnrm[s] = float(nrm);
      for (int j = 0; j < h.n; ++j) {
        int b = int((h.precursor - lmz[h.off + j]) * 100.f + 0.5f);
        if (b < 0 || b >= NBIN) continue;
        nlSpec[fill2[b]] = int32_t(s);
        nlW[fill2[b]++] = float(std::sqrt(std::max(0.f, lit[h.off + j])) / nrm);
      }
    }
    std::printf("neutral-loss index: %zu postings\n", nlSpec.size());
  }

  // Reverse-model bit logits per query (single-threaded: the model owns its buffers).
  int revNB = 0;
  if (haveRev) {
    rev::Model rm;
    if (!rm.load(dir + "/" + revPath)) { std::fprintf(stderr, "cannot load %s\n", revPath.c_str()); return 1; }
    revNB = rm.NB;
    std::vector<float> Pb;
    for (Query& q : Q) {
      std::vector<std::vector<std::pair<int, float>>> rows(q.spectra.size());
      for (size_t i = 0; i < q.spectra.size(); ++i) {
        const Head& h = head[q.spectra[i]];
        rev::Head rh{h.precursor, h.mol, 0, h.n, h.tag};
        rev::featurize(rh, mz.data() + h.off, it.data() + h.off, rows[i]);
      }
      int B = int(rows.size());
      rm.reserve(B);
      rm.setBatch(rows, false);
      rm.forward(B, false, 0.f, 0);
      rev::download(rm.P, Pb, int64_t(B) * revNB);
      q.logit.assign(revNB, 0.f);
      for (int b = 0; b < revNB; ++b) {
        float p = 0.f;
        for (int i = 0; i < B; ++i) p += Pb[size_t(i) * revNB + b];
        p = std::min(1.f - 1e-6f, std::max(1e-6f, p / float(B)));
        q.logit[b] = std::log(p / (1.f - p));
      }
    }
    std::printf("reverse model %s: %d bits\n", revPath.c_str(), revNB);
  }

  // candidates + fragment tables (the expensive part, done once)
  RDKit::PeriodicTable::getTable();
  std::mutex timeMu;
  double tStage1 = 0, tAnaTotal = 0;
  size_t nAnaSpec = 0;
  auto tEnum0 = std::chrono::steady_clock::now();
  std::atomic<size_t> next(0);
  std::vector<std::thread> th;
  for (unsigned t = 0; t < std::max(1u, std::thread::hardware_concurrency()); ++t)
    th.emplace_back([&]() {
      std::vector<float> acc(haveLib ? lhead.size() : 0, 0.f), acc2(haveLib ? lhead.size() : 0, 0.f);
      std::vector<uint8_t> mark(haveLib ? lhead.size() : 0, 0);
      std::vector<int32_t> touched;
      std::vector<float> qw;
      std::vector<std::pair<float, int32_t>> short1;
      std::vector<PairT> pairs;
      std::vector<uint8_t> usedQ, usedL;
      for (;;) {
        size_t qi = next++;
        if (qi >= Q.size()) break;
        Query& q = Q[qi];
        // Spectral analogs: library molecules whose spectra look like this
        // query's (any structure, precursor within 200 Da), top 80 by
        // similarity, with their fingerprints.
        // analogs[v] = (sim, fingerprint) of the top 80 molecules for variant v
        std::vector<std::pair<float, const std::vector<uint16_t>*>> analogs[NVAR];
        std::unordered_map<int, std::vector<uint16_t>> fpCache;
        if (haveLib) {
          auto t0 = std::chrono::steady_clock::now();
          double t1s = 0;
          std::unordered_map<int, float> molBest[NVAR];
          for (int s : q.spectra) {
            auto ta = std::chrono::steady_clock::now();
            const Head& h = head[s];
            double nrm = 0;
            for (int i = 0; i < h.n; ++i) nrm += std::max(0.f, it[h.off + i]);
            nrm = std::sqrt(nrm) + 1e-12;
            touched.clear();
            qw.resize(h.n);
            double qnNP2 = 0;
            for (int i = 0; i < h.n; ++i) {
              float wq = float(std::sqrt(std::max(0.f, it[h.off + i])) / nrm);
              qw[i] = wq;
              if (mz[h.off + i] <= h.precursor - 1.f) qnNP2 += double(wq) * wq;
              // fragment index: the peak's own bin and its nearer neighbour
              float x = mz[h.off + i] * 100.f;
              int b = int(x);
              int b2 = (x - float(b)) < 0.5f ? b - 1 : b + 1;
              for (int bb : {b, b2}) {
                if (bb < 0 || bb >= NBIN) continue;
                for (int64_t k = binStart[bb]; k < binStart[bb + 1]; ++k) {
                  int32_t ls = postSpec[k];
                  if (!mark[ls]) { mark[ls] = 1; touched.push_back(ls); }
                  acc[ls] += wq * postW[k];
                }
              }
              // neutral-loss index
              float y = (h.precursor - mz[h.off + i]) * 100.f + 0.5f;
              if (y < 0.f) continue;
              int c = int(y);
              int c2 = (y - float(c)) < 0.5f ? c - 1 : c + 1;
              for (int bb : {c, c2}) {
                if (bb < 0 || bb >= NBIN) continue;
                for (int64_t k = nlStart[bb]; k < nlStart[bb + 1]; ++k) {
                  int32_t ls = nlSpec[k];
                  if (!mark[ls]) { mark[ls] = 1; touched.push_back(ls); }
                  acc2[ls] += wq * nlW[k];
                }
              }
            }
            short1.clear();
            for (int32_t ls : touched) {
              float a1 = acc[ls], a2 = acc2[ls];
              acc[ls] = 0.f; acc2[ls] = 0.f; mark[ls] = 0;
              if (std::fabs(lhead[ls].precursor - h.precursor) > 200.f) continue;
              float sim = std::min(1.f, a1);
              if (sim >= 0.3f) { auto& mb = molBest[0][lhead[ls].mol]; mb = std::max(mb, sim); }
              if (a1 + a2 >= 0.2f) short1.emplace_back(a1 + a2, ls);
            }
            if (int(short1.size()) > shortK) {
              std::nth_element(short1.begin(), short1.begin() + shortK, short1.end(),
                               [](auto& a, auto& b) { return a.first > b.first; });
              short1.resize(shortK);
            }
            auto tb = std::chrono::steady_clock::now();
            t1s += std::chrono::duration<double>(tb - ta).count();
            // stage 2: exact (modified) cosine on the shortlist
            float qnNP = float(std::sqrt(qnNP2));
            for (auto& sl : short1) {
              int32_t ls = sl.second;
              const Head& lh = lhead[ls];
              McosOut o;
              modcos(mz.data() + h.off, qw.data(), h.n, h.precursor, qnNP, lmz.data() + lh.off,
                     lit.data() + lh.off, lh.n, lh.precursor, lnrm[ls], 0.01f, pairs, usedQ, usedL, o);
              float v[NVAR] = {0.f, o.cos, o.mcos, o.mcosNP, o.nMatch >= 6 ? o.mcos : 0.f,
                              o.nMatchNP >= 4 ? o.mcosNP : 0.f, o.nMatchNP >= 6 ? o.mcosNP : 0.f,
                              o.nMatchNP >= 8 ? o.mcosNP : 0.f, o.nMatchU >= 6 ? o.cos : 0.f};
              for (int vi = 1; vi < NVAR; ++vi)
                if (v[vi] >= 0.3f) { auto& mb = molBest[vi][lh.mol]; mb = std::max(mb, v[vi]); }
            }
          }
          for (int vi = 0; vi < NVAR; ++vi) {
            std::vector<std::pair<float, int>> top;
            for (auto& kv : molBest[vi]) top.emplace_back(kv.second, kv.first);
            std::sort(top.begin(), top.end(), [](auto& a, auto& b) { return a.first > b.first; });
            if (top.size() > 80) top.resize(80);
            for (auto& tp : top) {
              auto fc = fpCache.find(tp.second);
              if (fc == fpCache.end()) {
                std::unique_ptr<RDKit::RWMol> am;
                try { am.reset(RDKit::SmilesToMol(lsmiles[tp.second])); } catch (...) { am.reset(); }
                fc = fpCache.emplace(tp.second, am ? ecfp::compute(*am, 2, 2048) : std::vector<uint16_t>()).first;
              }
              if (fc->second.empty()) continue;
              analogs[vi].emplace_back(tp.first, &fc->second);
            }
          }
          double tt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
          {
            std::lock_guard<std::mutex> lk(timeMu);
            tStage1 += t1s; tAnaTotal += tt; nAnaSpec += q.spectra.size();
          }
        }
        std::set<int> cs;
        for (int s : q.spectra) {
          const Head& h = head[s];
          int a = adductIdx(h.tag);
          float shift = (a >= 0 && a < 10) ? ADDUCT_SHIFT[a] : 1.007276f;
          float neutral = h.precursor - shift;
          if (a == 8 || a == 9) neutral *= 0.5f;
          float w = neutral * ppm * 1e-6f;
          size_t lo = std::lower_bound(pmass.begin(), pmass.end(), neutral - w) - pmass.begin();
          size_t hi = std::upper_bound(pmass.begin(), pmass.end(), neutral + w) - pmass.begin();
          for (size_t k = lo; k < hi; ++k) cs.insert(pid[k]);
        }
        for (int c : cs) {
          std::unique_ptr<RDKit::RWMol> m;
          try { m.reset(RDKit::SmilesToMol(psmi[c])); } catch (...) { continue; }
          if (!m) continue;
          q.cand.push_back(c);
          q.frags.push_back(enumerate(*m));
          q.candN.push_back(int(m->getNumAtoms()));
          if (haveMlp) {
            auto nf = forward_model_nn::enumerateFragments(*m);
            float best = 0.f;
            for (int s : q.spectra) {
              const Head& h = head[s];
              forward_model_nn::Context ctx;
              int a = adductIdx(h.tag);
              ctx.adductIdx = a < forward_model_nn::Context::kNumAdducts ? a : -1;
              ctx.instrumentIdx = (h.tag < 0 || h.tag % 16 == 15) ? -1 : h.tag % 16;
              auto pred = forward_model_nn::scoreFragmentsWithContext(nf, mlp, ctx);
              best = std::max(best, mlpCosine(pred, mz.data() + h.off, it.data() + h.off, h.n));
            }
            q.mlpCos.push_back(best);
          }
          if (haveRev) {
            float r = 0.f;
            for (uint16_t b : ecfp::compute(*m, 2, revNB)) r += q.logit[b];
            q.revLL.push_back(r);
          }
          if (haveLib) {
            // sqrt-intensity cosine, each query peak to its best library peak within 0.01 Da
            float best = 0.f;
            auto ls = libSpecOf.find(pname[c]);
            if (ls != libSpecOf.end())
              for (int s : q.spectra) {
                const Head& h = head[s];
                double nq = 0;
                for (int i = 0; i < h.n; ++i) nq += it[h.off + i];
                for (int r : ls->second) {
                  const Head& lh = lhead[r];
                  double nl = 0, dot = 0;
                  for (int j = 0; j < lh.n; ++j) nl += lit[lh.off + j];
                  int j0 = 0;
                  for (int i = 0; i < h.n; ++i) {
                    double m = mz[h.off + i], bestI = 0;
                    while (j0 < lh.n && lmz[lh.off + j0] < m - 0.01) ++j0;
                    for (int j = j0; j < lh.n && lmz[lh.off + j] <= m + 0.01; ++j)
                      bestI = std::max(bestI, double(lit[lh.off + j]));
                    dot += std::sqrt(std::max(0.f, it[h.off + i]) * bestI);
                  }
                  if (nq > 0 && nl > 0) best = std::max(best, float(dot / std::sqrt(nq * nl)));
                }
              }
            q.libSim.push_back(best);
            // analog propagation: a candidate structurally close to a
            // library molecule whose spectrum matches is likely right
            auto cb = ecfp::compute(*m, 2, 2048);
            for (int vi = 0; vi < NVAR; ++vi) {
              float ap[3] = {0.f, 0.f, 0.f};
              for (auto& an : analogs[vi]) {
                const auto& af = *an.second;
                size_t i = 0, j = 0, inter = 0;
                while (i < cb.size() && j < af.size()) {
                  if (cb[i] == af[j]) { ++inter; ++i; ++j; }
                  else if (cb[i] < af[j]) ++i;
                  else ++j;
                }
                size_t uni = cb.size() + af.size() - inter;
                float tan = uni ? float(inter) / float(uni) : 0.f;
                float s2 = an.first * an.first;
                ap[0] = std::max(ap[0], tan * s2);
                ap[1] = std::max(ap[1], tan * s2 * an.first);
                ap[2] = std::max(ap[2], tan * s2 * s2);
              }
              for (int pi = 0; pi < 3; ++pi) q.anaV[vi][pi].push_back(ap[pi]);
            }
            q.anaP.push_back(q.anaV[anaVar][anaPowI].back());
          }
        }
        auto tp = poolIdx.find(q.name);
        if (tp != poolIdx.end())
          for (size_t i = 0; i < q.cand.size(); ++i) if (q.cand[i] == tp->second) q.truth = int(i);
        if (qi % 100 == 0) std::printf("  enumerated %zu / %zu\n", qi, Q.size());
      }
    });
  for (auto& t : th) t.join();
  if (haveLib)
    std::printf("analog search: %zu query spectra, stage1 %.1f thread-s, total (incl. stage2 + fingerprints) %.1f thread-s"
                " = %.1f ms/spectrum; enumeration wall %.1f s; K=%d\n",
                nAnaSpec, tStage1, tAnaTotal, 1000.0 * tAnaTotal / std::max<size_t>(1, nAnaSpec),
                std::chrono::duration<double>(std::chrono::steady_clock::now() - tEnum0).count(), shortK);
  size_t withTruth = 0, totC = 0;
  for (auto& q : Q) { withTruth += q.truth >= 0; totC += q.cand.size(); }
  std::printf("%zu queries, %.1f candidates each, truth present in %zu\n", Q.size(),
              double(totC) / Q.size(), withTruth);

  // MRR@25 over all queries (truth absent = 0), with ties broken pessimistically
  auto evaluate = [&](const Params& P, std::vector<std::vector<float>>* scoresOut) {
    std::vector<double> rr(Q.size(), 0.0);
    if (scoresOut) scoresOut->assign(Q.size(), {});
    std::atomic<size_t> nx(0);
    std::vector<std::thread> tt;
    for (unsigned t = 0; t < std::max(1u, std::thread::hardware_concurrency()); ++t)
      tt.emplace_back([&]() {
        std::vector<std::pair<double, double>> pred;
        for (;;) {
          size_t qi = nx++;
          if (qi >= Q.size()) break;
          const Query& q = Q[qi];
          std::vector<float> sc(q.cand.size(), 0.f);
          float eps = std::exp(P.logEps);
          for (size_t c = 0; c < q.cand.size(); ++c) {
            for (int s : q.spectra) {
              const Head& h = head[s];
              predict(q.frags[c], P, adductIdx(h.tag), q.candN[c], pred);
              sc[c] += loglik(pred, mz.data() + h.off, it.data() + h.off, h.n, 0.01f, eps, h.precursor);
            }
            sc[c] /= float(q.spectra.size());
          }
          if (q.truth >= 0) {
            int better = 0;
            for (size_t c = 0; c < sc.size(); ++c)
              if (int(c) != q.truth && sc[c] >= sc[q.truth]) ++better;
            if (better < 25) rr[qi] = 1.0 / (better + 1);
          }
          if (scoresOut) (*scoresOut)[qi] = std::move(sc);
        }
      });
    for (auto& t : tt) t.join();
    double s = 0;
    for (double x : rr) s += x;
    return s / Q.size();
  };

  Params P;
  if (mode == "fit") {
    // fit <...> <params_out> [params_init]: start from an earlier fit
    if (!outPath.empty() && P.load(dir + "/" + outPath)) std::printf("start from %s\n", outPath.c_str());
    double best = evaluate(P, nullptr);
    std::printf("initial MRR %.4f\n", best);
    // coordinate search: each parameter tried at +-step, step halves when a
    // full sweep finds nothing; E[INTACT] and the k weights included.
    float step = 1.0f;
    for (int sweep = 0; sweep < 12 && step > 0.05f; ++sweep) {
      bool improved = false;
      for (int i = 0; i < Params::N; ++i) {
        float s = (i == NCLS - 1 || i == NCLS) ? step * 0.1f : step;  // betas live on a smaller scale
        for (float d : {s, -s}) {
          Params T = P;
          *T.at(i) += d;
          if ((i == NCLS - 1 || i == NCLS) && *T.at(i) < 0.f) continue;
          double v = evaluate(T, nullptr);
          if (v > best + 1e-5) { best = v; P = T; improved = true; std::printf("  p%d -> %.3f  MRR %.4f\n", i, *P.at(i), best); break; }
        }
      }
      std::printf("sweep %d step %.3f MRR %.4f\n", sweep, step, best);
      std::fflush(stdout);
      if (!improved) step *= 0.5f;
    }
    P.save(dir + "/" + paramPath);
    std::printf("saved %s\n", paramPath.c_str());
    // ablations at the fitted point
    Params A = P; A.betaPos = 0.f; A.betaNeg = 0.f;
    std::printf("ablation: no charge retention %.4f\n", evaluate(A, nullptr));
    A = P; for (int i = 1; i < NCLS; ++i) A.E[i] = 1.0f;
    std::printf("ablation: flat bond classes %.4f\n", evaluate(A, nullptr));
    A = P; A.dofSlope = 0.f;
    std::printf("ablation: no size effect (dofSlope 0) %.4f\n", evaluate(A, nullptr));
    A = P; A.E[CGLYC] = A.E[ARYLC];
    std::printf("ablation: C-glycoside = aryl C-C %.4f\n", evaluate(A, nullptr));
    int nTruthCg = 0;
    for (auto& q : Q)
      if (q.truth >= 0)
        for (auto& fr : q.frags[q.truth])
          if (fr.cls == CGLYC) { ++nTruthCg; break; }
    std::printf("true structures with a C-glycosidic bond: %d / %zu\n", nTruthCg, Q.size());
    std::printf("fitted:");
    const char* cn[NCLS] = {"intact", "acyl", "glycosidic", "C-het", "aryl-C", "C-C", "ring", "ring-ar", "other", "C-glycosidic"};
    for (int i = 0; i < NCLS; ++i) std::printf(" %s=%.3f", cn[i], P.E[i]);
    std::printf(" E2=%.3f dofSlope=%.3f\n", P.E2, P.dofSlope);
    return 0;
  }

  if (!P.load(dir + "/" + paramPath)) std::printf("(params %s absent: defaults)\n", paramPath.c_str());
  std::vector<std::vector<float>> scores;
  double m = evaluate(P, &scores);
  // [features.txt]: per-candidate features for fuse_fit (learned weights).
  // "Q <name> <truth index or -1> <n>" then n lines
  // "phys mlp rev libSim anaP inLib name" (raw values; fuse_fit normalizes).
  std::string featPath = argc > 17 ? argv[17] : "";
  if (!featPath.empty()) {
    std::ofstream ff(dir + "/" + featPath);
    for (size_t qi = 0; qi < Q.size(); ++qi) {
      const Query& q = Q[qi];
      ff << "Q " << q.name << " " << q.truth << " " << q.cand.size() << "\n";
      for (size_t i = 0; i < q.cand.size(); ++i)
        ff << scores[qi][i] << " " << (haveMlp ? q.mlpCos[i] : 0.f) << " "
           << (haveRev ? q.revLL[i] : 0.f) << " " << (haveLib ? q.libSim[i] : 0.f) << " "
           << (haveLib ? q.anaP[i] : 0.f) << " "
           // "has reference spectra in THIS library", not the pool's train
           // flag: a held-out answer is a train structure whose spectra were
           // removed, so the pool flag marks it (1, with library sim 0) - a
           // signature the hidden test does not share. Fitted on that flag
           // the weights learned the holdout's construction (cqry 0.48 -> 0.77).
           << int(haveLib && libSpecOf.count(pname[q.cand[i]]) > 0) << " " << pname[q.cand[i]] << "\n";
    }
    std::printf("wrote features %s\n", featPath.c_str());
  }
  std::printf("MRR@25 %.4f\n", m);
  Params A = P; A.betaPos = 0.f; A.betaNeg = 0.f;
  std::printf("ablation: no charge retention %.4f\n", evaluate(A, nullptr));
  if (haveMlp || haveRev || haveLib) {
    // z(phys) + w z(mlp) + wr z(rev) + wl libSim; w < 0 means the MLP alone
    auto fused = [&](float w, float wr, float wl, std::vector<std::vector<float>>* out) {
      double tot = 0;
      // results go to a separate buffer: `out` is the caller's `scores`,
      // which this loop also READS as the physics term - clearing it first
      // (as v16-v19 did) left the output sorted on freed memory
      std::vector<std::vector<float>> res(out ? Q.size() : 0);
      for (size_t qi = 0; qi < Q.size(); ++qi) {
        auto zp = zscore(scores[qi]);
        std::vector<float> zm = haveMlp ? zscore(Q[qi].mlpCos) : std::vector<float>(zp.size(), 0.f);
        std::vector<float> zr = haveRev ? zscore(Q[qi].revLL) : std::vector<float>(zp.size(), 0.f);
        std::vector<float> f(zp.size());
        for (size_t i = 0; i < f.size(); ++i) {
          f[i] = w < 0.f ? zm[i] : zp[i] + w * zm[i] + wr * zr[i];
          // a sharp power: only a near-identical library spectrum (the same
          // molecule) should count - a linear term promotes every analog
          // whose spectrum is merely similar, which costs the de novo case
          if (haveLib) f[i] += wl * std::pow(Q[qi].libSim[i], libPow) + anaW * Q[qi].anaP[i];
        }
        int t = Q[qi].truth;
        if (t >= 0) {
          int better = 0;
          for (size_t i = 0; i < f.size(); ++i) if (int(i) != t && f[i] >= f[t]) ++better;
          if (better < 25) tot += 1.0 / (better + 1);
        }
        if (out) res[qi] = f;
      }
      if (out) *out = std::move(res);
      return tot / Q.size();
    };
    for (float wr : {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 5.f}) {
      if (!haveRev && wr > 0.f) break;
      for (float w : {-1.f, 0.f, 0.25f, 0.5f, 1.f}) {
        if (!haveMlp && w != 0.f) continue;
        std::printf("fusion w=%-5g wr=%-5g MRR %.4f\n", w, wr, fused(w, wr, 0.f, nullptr));
      }
    }
    if (haveLib) {
      libPow = 32.f;
      for (float wa : {0.f, 1.f, 2.f, 4.f, 8.f, 16.f}) {
        anaW = wa;
        for (float wl : {0.f, 8.f})
          std::printf("fusion w=0.25  wr=1  pow=32 wl=%-3g wa=%-4g MRR %.4f\n", wl, wa, fused(0.25f, 1.f, wl, nullptr));
      }
      // analog variants: one line per (variant, pow, wl), MRR over the wa grid
      const float WA[] = {0.f, 4.f, 8.f, 12.f, 16.f, 24.f, 32.f, 48.f};
      std::printf("ana grid (w=0.25 wr=1 libpow=32) wa:");
      for (float wa : WA) std::printf(" %6g", wa);
      std::printf("\n");
      for (int vi = 0; vi < NVAR; ++vi)
        for (int pi = 0; pi < 3; ++pi)
          for (float wl : {0.f, 8.f}) {
            for (auto& q : Q) q.anaP.swap(q.anaV[vi][pi]);
            std::printf("ana %-12s p=%d wl=%-2g", VARNAME[vi], pi + 2, wl);
            for (float wa : WA) { anaW = wa; std::printf(" %.4f", fused(0.25f, 1.f, wl, nullptr)); }
            std::printf("\n");
            for (auto& q : Q) q.anaP.swap(q.anaV[vi][pi]);
          }
      std::fflush(stdout);
    }
    libPow = fuseLPow;
    anaW = fuseA;
    if (!weightsPath.empty()) {
      // learned weights (fuse_fit, conditional logit) over the same features
      // fuse_fit reads: z(phys) z(mlp) z(rev) lib^32 lib^8 lib analog inLib
      std::ifstream wf(dir + "/" + weightsPath);
      double b[8] = {0};
      for (double& x : b) wf >> x;
      double tot = 0;
      std::vector<std::vector<float>> res(Q.size());
      for (size_t qi = 0; qi < Q.size(); ++qi) {
        const Query& q = Q[qi];
        auto zp = zscore(scores[qi]);
        std::vector<float> zm = haveMlp ? zscore(q.mlpCos) : std::vector<float>(zp.size(), 0.f);
        std::vector<float> zr = haveRev ? zscore(q.revLL) : std::vector<float>(zp.size(), 0.f);
        res[qi].resize(zp.size());
        for (size_t i = 0; i < zp.size(); ++i) {
          float l = haveLib ? std::max(0.f, q.libSim[i]) : 0.f;
          float inLib = (haveLib && libSpecOf.count(pname[q.cand[i]]) > 0) ? 1.f : 0.f;
          res[qi][i] = float(b[0] * zp[i] + b[1] * zm[i] + b[2] * zr[i] + b[3] * std::pow(l, 32.f) +
                             b[4] * std::pow(l, 8.f) + b[5] * l + b[6] * (haveLib ? q.anaP[i] : 0.f) +
                             b[7] * inLib);
        }
        if (q.truth >= 0) {
          int better = 0;
          for (size_t i = 0; i < res[qi].size(); ++i)
            if (int(i) != q.truth && res[qi][i] >= res[qi][q.truth]) ++better;
          if (better < 25) tot += 1.0 / (better + 1);
        }
      }
      scores = std::move(res);
      std::printf("output uses learned weights %s: MRR@25 %.4f\n", weightsPath.c_str(), tot / Q.size());
    } else if (fuseW > 0.f || fuseR > 0.f || fuseL > 0.f || fuseA > 0.f) {
      fused(fuseW, fuseR, fuseL, &scores);
      std::printf("output uses fusion w=%g wr=%g wl=%g\n", fuseW, fuseR, fuseL);
    }
  }
  if (!outPath.empty()) {
    std::ofstream out(dir + "/" + outPath);
    out << "molecule_id,smiles\n";
    for (size_t qi = 0; qi < Q.size(); ++qi) {
      std::vector<int> o(Q[qi].cand.size());
      for (size_t i = 0; i < o.size(); ++i) o[i] = int(i);
      std::sort(o.begin(), o.end(), [&](int a, int b) { return scores[qi][a] > scores[qi][b]; });
      out << Q[qi].name << ",";
      for (size_t i = 0; i < o.size() && i < 25; ++i)
        out << (i ? ";" : "") << (writeSmiles ? psmi[Q[qi].cand[o[i]]] : pname[Q[qi].cand[o[i]]]);
      out << "\n";
    }
  }
  return 0;
}
