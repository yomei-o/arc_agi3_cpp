// phys_pc: phys_gen + a whole-PubChem candidate channel merged into the final
// list (the top public notebooks' "PubChem slots", here by calibrated score).
//
// Extra keys in the options string (argv[24]):
//   pc=1          enable; tables pc_mass.bin / pc_off.bin / pc_smi.bin /
//                 pc_pop.bin (pc_build.cpp, pop_build.cpp) in pcdir (default <dir>)
//   pcdir=path    directory of the pc_*.bin tables (absolute)
//   pcpre=N       prefilter: per query, every PubChem compound in the +-ppm
//                 windows is parsed and scored rev + pcpp*pop (REV2 vocabulary
//                 bits, popularity); the top N go on (default 300)
//   pcpp=x        popularity weight in the prefilter (default 20: popularity first,
//                 REV2 breaks its ties - see pcslot_STATUS.md)
//   pcn=N         after dropping stereo-blind duplicates of pool / generated
//                 candidates (and of each other), the top N get the full
//                 features of the main list (physics, MLP, analogs) (default 50)
//   pcpop=x       popularity weight in the PubChem list's final score
//                 (fused hand score with z-scores taken on the MAIN list's
//                 mean/std + pcpop * pop) (default 2)
//   pck=K         PubChem candidates inserted (default 2)
//   gate=x        insert only when the best library similarity among the main
//                 candidates (libSim) is below x (default 0.9)
//   gmarg=x       ... and the main list's top1 - top2 fused margin is below x (default 1e9)
//   pcmode=score  (default) merge by score: a pick enters where its
//                 fused - pcpop*pop + pcsw*pop + pcbias beats the next main score
//                 (main scores taken before the FPNet term, which picks lack),
//                 never above position pcmin (defaults pcsw 1, pcbias -7, pcmin 2;
//                 pcmin=1 = the aggressive variant, see pcslot_STATUS.md)
//   pcmode=slot   the public notebooks' fixed slots instead:
//   slots=a:b:..  1-based positions of the final list the picks take (default 2:4:6:8:10)
//   pcout=file    the inserted list (CSV, like out.csv); out.csv stays the plain list.
//                 Without pcout, out.csv gets the inserted list (production)
//   pcdet=file    per query gate stats + every fully scored PubChem candidate
//   pcall=file    every window candidate (name cid rev pop truth) - recall study
//   pcrecall=1    find the truth among ALL window candidates (formula match, then
//                 stereo-blind key) and report recall / prefilter rank
// In eval output a PubChem candidate equal to the truth is written under the
// truth's name, others as P<cid>; with `smiles`, as its PubChem SMILES.
//
// phys_gen: phys_ana3 + structure generation from spectral analogs.
//
// Same arguments as phys_ana3 plus one options string after [wf] (argv[24]),
// comma-separated key=value:
//   gen=0|1      generate candidates: the top `nan` analogs (variant `gvar`,
//                default 4 = modified cosine with >= 6 matched peaks) whose
//                precursor-mass difference matches a transformation (or a
//                2-step chain) of gen_ops.h are edited at every valid site;
//                products are deduped (non-isomeric canonical SMILES) against
//                each other and the pool window, mass-checked against the
//                window, ordered by (source sim desc, steps, rank) and capped
//   rm=0|1       evaluation: remove the true structure (and stereo-blind
//                duplicates) from the pool window - "answer not in pool"
//   cap=N        generated candidates per query (default 1000)
//   nan=N        analogs used for generation (default 80)
//   two=0|1      two-step chains (default 1)
//   prior=file   per-op log prior ("op logprior" lines) for the op-prior feature
//   gw=a:b:...   hand weights for the generation features (added to the hand
//                fusion); a learned weights file ([weights.txt]) with 8+NG
//                numbers supplies them instead
//   det=file     per generated candidate details (query key truth sim rank steps nsup nsib chains)
//   thr=N        threads (default: all cores)
//   grid=0       skip the fusion grids
// Generation features (extra columns after the name in [features.txt]):
//   isGen, isGen*sim^3, isGen*(steps==2), isGen*log1p(nSupport),
//   isGen*log1p(nSib), isGen*opPrior
// Output: generated candidates are written as SMILES with `smiles`; without,
// a generated candidate equal to the truth is written under the truth's name
// (others as G<k>), so mrr.py scores the CSV.
//
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
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <GraphMol/GraphMol.h>
#include <GraphMol/PeriodicTable.h>
#include <GraphMol/RingInfo.h>
#include <GraphMol/SmilesParse/SmilesParse.h>

#include "forward_model.h"
#include "forward_model_nn.h"
#include "rev/ecfp.h"  // copies of ../rev headers (the dataset ships them inside rdkit-min/)
#include "rev/rev_model.hpp"
#include "rev/fp2.h"           // REV2 models: fp2 vocabulary fingerprints
#include "rev/rev2_model.hpp"
#include "gen_ops.h"
#include <array>
#include <map>
#include <sstream>

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
  // generation: cand[i] < 0 is generated candidate gi = -cand[i] - 1
  std::vector<std::string> gSmi;
  std::vector<std::array<float, 6>> gF;  // per candidate (0 for pool candidates)
  std::vector<float> phys;               // physics score computed in the enumeration pass (eval)
 std::string tsmi;                      // true structure's SMILES (evaluation; may be empty)
  int nGenRaw = 0;                       // distinct products before pool dedupe / cap
  bool truthGenRaw = false, truthGenKept = false;
  std::string det;                       // generated-candidate detail lines
  // PubChem channel
  struct PcCand {
    int32_t cid; std::string smi; bool truth;
    float rev, pop, phys, mlp, ana, fused;
    float zp = 0, zm = 0, zr = 0;
  };
  std::vector<PcCand> pc;                // fully scored, sorted by fused desc
  float libMax = 0.f, top1 = 0.f, top2 = 0.f, anaTop = 0.f;
  std::vector<float> topS;               // final scores of the main list's top 25
  int pcWin = 0, pcTruthWin = -1, pcTruthPre = -1;  // window size; truth's rank in window (rev+pop) / -1
  bool pcTruthInWin = false, pcTruthKept = false;
  std::string pcAll;
};
constexpr int NG = 6;

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
    std::fprintf(stderr, "usage: phys_ana3 fit|eval <dir> <qry> <pool_stem> <ppm> <params> [out.csv] ... [fuse_w.txt] [fpnet_scores.txt] [wf]\n");
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
  // phys_fp: [fpnet_scores.txt] [wf=1]: "<molecule_id>\t<pool name>\t<score>" lines (fpnet_pool),
  // added to the final fused score as + wf * z(fpnet) (per-query z over its candidates;
  // a candidate without a score gets the query's mean, i.e. z = 0)
  std::string fpPath = argc > 22 ? argv[22] : "";
  float fuseF = argc > 23 ? float(std::atof(argv[23])) : 1.0f;
  // generation options (see header)
  std::map<std::string, std::string> opt;
  if (argc > 24) {
    std::stringstream ss(argv[24]);
    std::string kv;
    while (std::getline(ss, kv, ',')) {
      size_t e = kv.find('=');
      if (e != std::string::npos) opt[kv.substr(0, e)] = kv.substr(e + 1);
    }
  }
  auto optI = [&](const char* k, int d) { auto i = opt.find(k); return i == opt.end() ? d : std::atoi(i->second.c_str()); };
  const bool doGen = optI("gen", 0) != 0, rmTruth = optI("rm", 0) != 0, twoStep = optI("two", 1) != 0;
  const bool doGrid = optI("grid", 1) != 0;
  const int genCap = optI("cap", 1000), genNan = optI("nan", 80), genVar = optI("gvar", 4);
  const unsigned nThr = unsigned(optI("thr", int(std::max(1u, std::thread::hardware_concurrency()))));
  std::vector<double> gw(NG, 0.0);
  if (opt.count("gw")) {
    std::stringstream ss(opt["gw"]);
    std::string v;
    for (int k = 0; k < NG && std::getline(ss, v, ':'); ++k) gw[k] = std::atof(v.c_str());
  }
  std::unordered_map<std::string, float> opPrior;
  float opPriorDefault = 0.f;
  if (opt.count("prior")) {
    std::ifstream pf(dir + "/" + opt["prior"]);
    std::string k;
    float v;
    while (pf >> k >> v) {
      if (k == "default") opPriorDefault = v;
      else opPrior[k] = v;
    }
    std::printf("op priors: %zu (default %g)\n", opPrior.size(), opPriorDefault);
  }
  // a chain's prior: sum of its ops' single-op priors
  auto chainPrior = [&](const std::string& ch) {
    float s = 0.f;
    std::stringstream ss(ch);
    std::string o;
    while (std::getline(ss, o, '/')) {
      auto f = opPrior.find(o);
      s += f == opPrior.end() ? opPriorDefault : f->second;
    }
    return s;
  };
  auto optF = [&](const char* k, double d) { auto i = opt.find(k); return i == opt.end() ? d : std::atof(i->second.c_str()); };
  const bool doPc = optI("pc", 0) != 0, pcRecall = optI("pcrecall", 0) != 0;
  const int pcPre = optI("pcpre", 300), pcN = optI("pcn", 50), pcK = optI("pck", 2);
  const float pcPP = float(optF("pcpp", 20.0)), pcPop = float(optF("pcpop", 2.0));
  const float pcGate = float(optF("gate", 0.9)), pcGMarg = float(optF("gmarg", 1e9));
  // pcmode=score: instead of fixed slots, a pick enters where its score
  // (fused without popularity + pcsw * pop + pcbias) beats the main list's
  // next score, never above position pcmin
  const bool pcScoreMode = !(opt.count("pcmode") && opt["pcmode"] == "slot");
  const float pcSW = float(optF("pcsw", 1.0)), pcBias = float(optF("pcbias", -7.0));
  const int pcMin = optI("pcmin", 2);
  std::vector<int> pcSlots;
  {
    std::stringstream ss(opt.count("slots") ? opt["slots"] : std::string("2:4:6:8:10"));
    std::string v;
    while (std::getline(ss, v, ':')) if (!v.empty()) pcSlots.push_back(std::atoi(v.c_str()));
  }
  const std::string pcDir = opt.count("pcdir") ? opt["pcdir"] : dir;
  std::vector<float> pcMass, pcPopV;
  std::vector<uint64_t> pcOff;
  if (doPc) {
    auto rd = [&](const std::string& p, auto& v) {
      std::vector<char> b = slurp(pcDir + "/" + p);
      v.resize(b.size() / sizeof(v[0]));
      std::memcpy(v.data(), b.data(), v.size() * sizeof(v[0]));
    };
    rd("pc_mass.bin", pcMass);
    rd("pc_off.bin", pcOff);
    rd("pc_pop.bin", pcPopV);
    std::printf("pubchem: %zu compounds; prefilter %d (pop %g), full %d, insert %d", pcMass.size(), pcPre, pcPP, pcN, pcK);
    if (pcScoreMode) {
      std::printf(" by score (pop %g, bias %g, from position %d)", pcSW, pcBias, pcMin);
    } else {
      std::printf(" at slots");
      for (int s : pcSlots) std::printf(" %d", s);
    }
    std::printf(", gate libSim < %g, margin < %g, list pop %g\n", pcGate, pcGMarg, pcPop);
  }
  if (doGen) std::printf("generation: cap %d, analogs %d (variant %d), two-step %d, remove truth %d, threads %u\n",
                         genCap, genNan, genVar, int(twoStep), int(rmTruth), nThr);
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
  std::vector<std::string> qnames, qsmiles;
  {
    std::ifstream f(dir + "/" + qs + "_mols.txt");
    std::string line;
    while (std::getline(f, line)) {
      if (!line.empty() && line.back() == '\r') line.pop_back();
      size_t t = line.find('\t');
      qnames.push_back(line.substr(0, t));
      qsmiles.push_back(t == std::string::npos ? "" : line.substr(t + 1));
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
    q.tsmi = qsmiles[kv.first];
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
  // REV2 model files (magic "REV2") carry their own input layout and an fp2
  // vocabulary defining the output bits; candidate bits then come from revVocab.
  bool rev2Mode = false;
  fp2::Vocab revVocab;
  if (haveRev) {
    int32_t magic = 0;
    {
      std::ifstream mf(dir + "/" + revPath, std::ios::binary);
      mf.read(reinterpret_cast<char*>(&magic), 4);
    }
    rev2Mode = magic == 0x32564552;
  }
  if (haveRev && rev2Mode) {
    rev2::Model rm;
    if (!rm.load(dir + "/" + revPath)) { std::fprintf(stderr, "cannot load %s\n", revPath.c_str()); return 1; }
    if (rm.vocab.empty() || int(rm.vocab.size()) != rm.NB) {
      std::fprintf(stderr, "%s: REV2 model without embedded vocabulary (nVocab %zu, NB %d)\n", revPath.c_str(), rm.vocab.size(), rm.NB);
      return 1;
    }
    revNB = rm.NB;
    revVocab.rE = rm.rE; revVocab.rF = rm.rF; revVocab.hashes = rm.vocab; revVocab.build();
    // logit of the mean p, clipped to [0.01, 0.99] (REV2_NOCLIP=1: [1e-6, 1-1e-6] as v1)
    const char* nc = std::getenv("REV2_NOCLIP");
    const float lo = (nc && *nc == '1') ? 1e-6f : 0.01f;
    std::vector<float> Pb;
    for (Query& q : Q) {
      std::vector<std::vector<std::pair<int, float>>> rows(q.spectra.size());
      for (size_t i = 0; i < q.spectra.size(); ++i) {
        const Head& h = head[q.spectra[i]];
        rev2::Head rh{h.precursor, h.mol, 0, h.n, h.tag};
        rev2::featurize(rm.fc, rh, mz.data() + h.off, it.data() + h.off, rows[i]);
      }
      int B = int(rows.size());
      rm.reserve(B);
      rm.setBatch(rows, false);
      rm.forward(B, false, 0.f, 0);
      rev2::download(rm.P, Pb, int64_t(B) * revNB);
      q.logit.assign(revNB, 0.f);
      for (int b = 0; b < revNB; ++b) {
        float p = 0.f;
        for (int i = 0; i < B; ++i) p += Pb[size_t(i) * revNB + b];
        p = std::min(1.f - lo, std::max(lo, p / float(B)));
        q.logit[b] = std::log(p / (1.f - p));
      }
    }
    std::printf("reverse model %s: REV2, %d vocab bits (ECFP r%d + FCFP r%d), clip %g\n", revPath.c_str(), revNB, rm.rE, rm.rF, lo);
  } else if (haveRev) {
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
  double tGen = 0, tPc = 0;
  std::atomic<long> nRmDup(0), nTruthDup(0), nOutWin(0);
  const std::string detPath = opt.count("det") ? opt["det"] : "";
  const bool keepFrags = mode == "fit";
  Params P0;
  if (!keepFrags && !P0.load(dir + "/" + paramPath)) std::printf("(params %s absent: defaults)\n", paramPath.c_str());
  size_t nAnaSpec = 0;
  auto tEnum0 = std::chrono::steady_clock::now();
  std::atomic<size_t> next(0);
  std::vector<std::thread> th;
  for (unsigned t = 0; t < nThr; ++t)
    th.emplace_back([&]() {
      std::vector<float> acc(haveLib ? lhead.size() : 0, 0.f), acc2(haveLib ? lhead.size() : 0, 0.f);
      std::vector<uint8_t> mark(haveLib ? lhead.size() : 0, 0);
      std::vector<int32_t> touched;
      std::vector<float> qw;
      std::vector<std::pair<float, int32_t>> short1;
      std::vector<PairT> pairs;
      std::vector<uint8_t> usedQ, usedL;
      std::vector<std::pair<double, double>> predBuf;
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
        std::vector<std::pair<float, int>> genAn;  // (sim, library molecule) for generation
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
            if (vi == genVar)
              for (size_t k = 0; k < top.size() && int(k) < genNan; ++k) genAn.push_back(top[k]);
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
        // neutral-mass windows of this query (one per spectrum / adduct)
        std::vector<std::pair<double, double>> win;
        std::set<int> cs;
        for (int s : q.spectra) {
          const Head& h = head[s];
          int a = adductIdx(h.tag);
          float shift = (a >= 0 && a < 10) ? ADDUCT_SHIFT[a] : 1.007276f;
          float neutral = h.precursor - shift;
          if (a == 8 || a == 9) neutral *= 0.5f;
          float w = neutral * ppm * 1e-6f;
          win.emplace_back(double(neutral) - w, double(neutral) + w);
          size_t lo = std::lower_bound(pmass.begin(), pmass.end(), neutral - w) - pmass.begin();
          size_t hi = std::upper_bound(pmass.begin(), pmass.end(), neutral + w) - pmass.begin();
          for (size_t k = lo; k < hi; ++k) cs.insert(pid[k]);
        }
        // stereo-blind key: non-isomeric canonical SMILES, re-parsed once so
        // pool, truth and generated structures go through the same path
        auto canon = [](const RDKit::ROMol& m) -> std::string {
          std::string k = gen::key(m);
          std::unique_ptr<RDKit::RWMol> r;
          try { r.reset(RDKit::SmilesToMol(k)); } catch (...) { r.reset(); }
          return r ? gen::key(*r) : k;
        };
        const bool needKeys = doGen || rmTruth || doPc;
        std::string tkey;
        if (needKeys && !q.tsmi.empty()) {
          std::unique_ptr<RDKit::RWMol> tm;
          try { tm.reset(RDKit::SmilesToMol(q.tsmi)); } catch (...) { tm.reset(); }
          if (tm) tkey = canon(*tm);
        }
        int truthPool = -1;
        {
          auto tp = poolIdx.find(q.name);
          if (tp != poolIdx.end()) truthPool = tp->second;
        }
        std::vector<std::pair<std::unique_ptr<RDKit::RWMol>, int>> cl;  // (molecule, candidate code)
        std::unordered_map<std::string, int> poolKeys;
        bool truthInPool = false;
        for (int c : cs) {
          if (rmTruth && c == truthPool) continue;
          std::unique_ptr<RDKit::RWMol> m;
          try { m.reset(RDKit::SmilesToMol(psmi[c])); } catch (...) { continue; }
          if (!m) continue;
          if (needKeys) {
            std::string k = canon(*m);
            if (rmTruth && !tkey.empty() && k == tkey) { ++nRmDup; continue; }  // stereo-blind duplicate of the answer
            poolKeys.emplace(k, c);
          }
          if (c == truthPool) truthInPool = true;
          cl.emplace_back(std::move(m), c);
        }
        std::vector<std::array<float, NG>> gfeat(cl.size(), std::array<float, NG>{0, 0, 0, 0, 0, 0});
        if (doGen && !genAn.empty()) {
          auto tg0 = std::chrono::steady_clock::now();
          std::unordered_map<std::string, gen::Prod> prod;
          std::vector<std::vector<int>> chains;
          for (size_t r = 0; r < genAn.size(); ++r) {
            std::unique_ptr<RDKit::RWMol> am;
            try {
              std::unique_ptr<RDKit::RWMol> a0(RDKit::SmilesToMol(lsmiles[genAn[r].second]));
              if (a0) am.reset(RDKit::SmilesToMol(gen::key(*a0)));  // drop stereo
            } catch (...) { am.reset(); }
            if (!am) continue;
            gen::chainsFor(gen::molMass(*am), win, chains, twoStep);
            if (chains.empty()) continue;
            try {
              gen::generateFrom(*am, int(r), genAn[r].first, chains, prod, 2000);
            } catch (...) {
            }
          }
          // canonical re-key, merge, pool dedupe, mass check
          std::unordered_map<std::string, gen::Prod*> byKey;
          for (auto& kv : prod) {
            std::string k = canon(*kv.second.mol);
            auto ins = byKey.emplace(k, &kv.second);
            if (!ins.second) {
              gen::Prod& a = *ins.first->second;
              gen::Prod& b = kv.second;
              if (b.sim > a.sim || (b.sim == a.sim && b.nSib < a.nSib)) { a.sim = b.sim; a.nSib = b.nSib; }
              a.rank = std::min(a.rank, b.rank);
              a.steps = std::min(a.steps, b.steps);
              for (int s : b.sup) if (std::find(a.sup.begin(), a.sup.end(), s) == a.sup.end()) a.sup.push_back(s);
              for (auto& s : b.chains) if (std::find(a.chains.begin(), a.chains.end(), s) == a.chains.end()) a.chains.push_back(s);
            }
          }
          q.nGenRaw = int(byKey.size());
          q.truthGenRaw = !tkey.empty() && byKey.count(tkey) > 0;
          std::vector<std::pair<const std::string*, gen::Prod*>> items;
          for (auto& kv : byKey) {
            if (poolKeys.count(kv.first)) continue;
            if (truthInPool && kv.first == tkey) { ++nTruthDup; continue; }  // the answer's own pool entry stays
            double pm = gen::molMass(*kv.second->mol);
            bool inw = false;
            for (auto& w : win) if (pm >= w.first - 1e-4 && pm <= w.second + 1e-4) inw = true;
            if (!inw) { ++nOutWin; continue; }
            items.emplace_back(&kv.first, kv.second);
          }
          std::sort(items.begin(), items.end(), [](auto& a, auto& b) {
            if (a.second->sim != b.second->sim) return a.second->sim > b.second->sim;
            if (a.second->steps != b.second->steps) return a.second->steps < b.second->steps;
            if (a.second->rank != b.second->rank) return a.second->rank < b.second->rank;
            return *a.first < *b.first;
          });
          if (int(items.size()) > genCap) items.resize(genCap);
          std::ostringstream det;
          for (auto& itm : items) {
            std::unique_ptr<RDKit::RWMol> gm;
            try { gm.reset(RDKit::SmilesToMol(*itm.first)); } catch (...) { gm.reset(); }
            if (!gm) continue;
            const gen::Prod& p = *itm.second;
            int gi = int(q.gSmi.size());
            q.gSmi.push_back(*itm.first);
            float pr = -1e9f;
            std::string chs;
            for (auto& ch : p.chains) { pr = std::max(pr, chainPrior(ch)); chs += (chs.empty() ? "" : ";") + ch; }
            gfeat.push_back({1.f, p.sim * p.sim * p.sim, p.steps == 2 ? 1.f : 0.f, std::log1p(float(p.sup.size())),
                             std::log1p(float(p.nSib)), opPrior.empty() ? 0.f : pr});
            cl.emplace_back(std::move(gm), -(gi + 1));
            bool isT = !tkey.empty() && *itm.first == tkey;
            if (isT) q.truthGenKept = true;
            if (!detPath.empty())
              det << q.name << "\t" << *itm.first << "\t" << int(isT) << "\t" << p.sim << "\t" << p.rank << "\t"
                  << p.steps << "\t" << p.sup.size() << "\t" << p.nSib << "\t" << chs << "\n";
          }
          q.det = det.str();
          double tg = std::chrono::duration<double>(std::chrono::steady_clock::now() - tg0).count();
          std::lock_guard<std::mutex> lk(timeMu);
          tGen += tg;
        }
        // physics / forward-MLP / reverse-model scores of one structure
        auto physOf = [&](const RDKit::ROMol& m) {
          auto fr = enumerate(m);
          float eps = std::exp(P0.logEps), sc = 0.f;
          for (int s : q.spectra) {
            const Head& h = head[s];
            predict(fr, P0, adductIdx(h.tag), int(m.getNumAtoms()), predBuf);
            sc += loglik(predBuf, mz.data() + h.off, it.data() + h.off, h.n, 0.01f, eps, h.precursor);
          }
          return sc / float(q.spectra.size());
        };
        auto mlpOf = [&](const RDKit::ROMol& m) {
          auto nf = forward_model_nn::enumerateFragments(m);
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
          return best;
        };
        auto revOf = [&](const RDKit::ROMol& m) {
          float r = 0.f;
          if (rev2Mode) {
            for (uint16_t b : revVocab.bits(m)) r += q.logit[b];
          } else {
            for (uint16_t b : ecfp::compute(m, 2, revNB)) r += q.logit[b];
          }
          return r;
        };
        // max_a Tanimoto(cand, a) * sim_a^(2,3,4) over the analogs of variant vi
        auto anaOf = [&](const std::vector<uint16_t>& cb, int vi, float ap[3]) {
          ap[0] = ap[1] = ap[2] = 0.f;
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
        };
        for (size_t ci = 0; ci < cl.size(); ++ci) {
          RDKit::RWMol* m = cl[ci].first.get();
          const int c = cl[ci].second;
          q.cand.push_back(c);
          q.gF.push_back(gfeat[ci]);
          if (keepFrags) {
            q.frags.push_back(enumerate(*m));
          } else {
            // physics score now; the fragment table is not kept (memory)
            q.phys.push_back(physOf(*m));
          }
          q.candN.push_back(int(m->getNumAtoms()));
          if (haveMlp) q.mlpCos.push_back(mlpOf(*m));
          if (haveRev) q.revLL.push_back(revOf(*m));
          if (haveLib) {
            // sqrt-intensity cosine, each query peak to its best library peak within 0.01 Da
            float best = 0.f;
            auto ls = c >= 0 ? libSpecOf.find(pname[c]) : libSpecOf.end();
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
                    double mm = mz[h.off + i], bestI = 0;
                    while (j0 < lh.n && lmz[lh.off + j0] < mm - 0.01) ++j0;
                    for (int j = j0; j < lh.n && lmz[lh.off + j] <= mm + 0.01; ++j)
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
              float ap[3];
              anaOf(cb, vi, ap);
              for (int pi = 0; pi < 3; ++pi) q.anaV[vi][pi].push_back(ap[pi]);
            }
            q.anaP.push_back(q.anaV[anaVar][anaPowI].back());
          }
        }
        // ---- PubChem channel ------------------------------------------------
        if (doPc && haveRev && !keepFrags) {
          auto tp0 = std::chrono::steady_clock::now();
          for (float l : q.libSim) q.libMax = std::max(q.libMax, l);
          if (!genAn.empty()) q.anaTop = genAn[0].first;
          // the truth's formula signature (recall study)
          auto sigOf = [](const RDKit::ROMol& m) {
            std::map<int, int> cnt;
            int chg = 0;
            for (const auto a : m.atoms()) { ++cnt[a->getAtomicNum()]; cnt[1] += int(a->getTotalNumHs(false)); chg += a->getFormalCharge(); }
            std::string s;
            for (auto& kv : cnt) s += std::to_string(kv.first) + ":" + std::to_string(kv.second) + ",";
            return s + "q" + std::to_string(chg);
          };
          std::string tsig;
          if (pcRecall && !tkey.empty()) {
            std::unique_ptr<RDKit::RWMol> tm;
            try { tm.reset(RDKit::SmilesToMol(tkey)); } catch (...) { tm.reset(); }
            if (tm) tsig = sigOf(*tm);
          }
          // window ranges in the mass-sorted table (merged; one contiguous read each)
          std::vector<std::pair<size_t, size_t>> rng;
          for (auto& w : win) {
            size_t lo = std::lower_bound(pcMass.begin(), pcMass.end(), float(w.first)) - pcMass.begin();
            size_t hi = std::upper_bound(pcMass.begin(), pcMass.end(), float(w.second)) - pcMass.begin();
            if (hi > lo) rng.emplace_back(lo, hi);
          }
          std::sort(rng.begin(), rng.end());
          std::vector<std::pair<size_t, size_t>> mr;
          for (auto& r : rng) {
            if (!mr.empty() && r.first <= mr.back().second) mr.back().second = std::max(mr.back().second, r.second);
            else mr.push_back(r);
          }
          struct W { size_t k; float rev, pop, pre; int32_t cid; std::string smi; bool truth; };
          std::vector<W> all;
          std::ifstream fs(pcDir + "/pc_smi.bin", std::ios::binary);
          std::ifstream fc(pcDir + "/pc_cid.bin", std::ios::binary);
          std::string buf;
          std::vector<int32_t> cidb;
          for (auto& r : mr) {
            uint64_t o0 = pcOff[r.first], o1 = pcOff[r.second];
            buf.resize(size_t(o1 - o0));
            fs.seekg(std::streamoff(o0));
            fs.read(buf.data(), std::streamsize(o1 - o0));
            cidb.resize(r.second - r.first);
            if (fc) {  // optional (the Kaggle dataset has no pc_cid.bin): else the table index
              fc.seekg(std::streamoff(r.first * 4));
              fc.read(reinterpret_cast<char*>(cidb.data()), std::streamsize(cidb.size() * 4));
            } else {
              for (size_t k = r.first; k < r.second; ++k) cidb[k - r.first] = int32_t(k);
            }
            for (size_t k = r.first; k < r.second; ++k) {
              std::string smi = buf.substr(size_t(pcOff[k] - o0), size_t(pcOff[k + 1] - pcOff[k]));
              std::unique_ptr<RDKit::RWMol> m;
              try { m.reset(RDKit::SmilesToMol(smi)); } catch (...) { m.reset(); }
              if (!m) continue;
              W w{k, revOf(*m), pcPopV.empty() ? 0.f : pcPopV[k], 0.f, cidb[k - r.first], std::move(smi), false};
              w.pre = w.rev + pcPP * w.pop;
              if (!tsig.empty() && sigOf(*m) == tsig && canon(*m) == tkey) { w.truth = true; q.pcTruthInWin = true; }
              all.push_back(std::move(w));
            }
          }
          q.pcWin = int(all.size());
          std::sort(all.begin(), all.end(), [](const W& a, const W& b) { return a.pre != b.pre ? a.pre > b.pre : a.k < b.k; });
          for (size_t i = 0; i < all.size(); ++i) if (all[i].truth) { q.pcTruthWin = int(i); break; }
          if (opt.count("pcall")) {
            std::ostringstream os;
            for (auto& w : all) os << q.name << "\t" << w.cid << "\t" << w.rev << "\t" << w.pop << "\t" << int(w.truth) << "\n";
            q.pcAll = os.str();
          }
          if (int(all.size()) > pcPre) all.resize(pcPre);
          // stereo-blind dedupe against the main list and each other
          std::unordered_set<std::string> seen;
          for (auto& kv : poolKeys) seen.insert(kv.first);
          for (auto& g : q.gSmi) seen.insert(g);
          // main-list z statistics: PubChem candidates go on the same scale
          auto msd = [](const std::vector<float>& v, double& m, double& s) {
            m = 0; s = 0;
            for (float x : v) m += x;
            m /= double(std::max<size_t>(1, v.size()));
            for (float x : v) s += (x - m) * (x - m);
            s = std::sqrt(s / double(std::max<size_t>(1, v.size()))) + 1e-6;
          };
          double mp, sp, mm = 0, sm = 1, mrv, srv;
          msd(q.phys, mp, sp);
          if (haveMlp) msd(q.mlpCos, mm, sm);
          msd(q.revLL, mrv, srv);
          const bool ownStats = q.cand.size() < 2;
          std::vector<Query::PcCand> got;
          int pre = 0;
          for (auto& w : all) {
            ++pre;
            if (int(got.size()) >= pcN) break;
            std::unique_ptr<RDKit::RWMol> m;
            try { m.reset(RDKit::SmilesToMol(w.smi)); } catch (...) { m.reset(); }
            if (!m) continue;
            std::string k = canon(*m);
            if (!seen.insert(k).second) continue;
            bool isT = !tkey.empty() && k == tkey;
            if (isT && truthInPool) continue;  // the answer's pool entry (canonicalized differently) stays the only copy
            if (isT) { q.pcTruthKept = true; q.pcTruthPre = pre - 1; }
            Query::PcCand pcand{w.cid, w.smi, isT, w.rev, w.pop, physOf(*m), haveMlp ? mlpOf(*m) : 0.f, 0.f, 0.f};
            if (haveLib) {
              auto cb = ecfp::compute(*m, 2, 2048);
              float ap[3];
              anaOf(cb, anaVar, ap);
              pcand.ana = ap[anaPowI];
            }
            got.push_back(std::move(pcand));
          }
          if (ownStats && !got.empty()) {
            std::vector<float> a, b, c;
            for (auto& g : got) { a.push_back(g.phys); b.push_back(g.mlp); c.push_back(g.rev); }
            msd(a, mp, sp); if (haveMlp) msd(b, mm, sm); msd(c, mrv, srv);
          }
          for (auto& g : got) {
            double zp = (g.phys - mp) / sp, zm = haveMlp ? (g.mlp - mm) / sm : 0.0, zr = (g.rev - mrv) / srv;
            // the hand fusion of the main list (libSim = 0: not a library molecule) + popularity
            g.fused = float(zp + fuseW * zm + fuseR * zr + fuseA * g.ana + pcPop * g.pop);
            g.zp = float(zp); g.zm = float(zm); g.zr = float(zr);
          }
          std::stable_sort(got.begin(), got.end(), [](auto& a, auto& b) { return a.fused > b.fused; });
          q.pc = std::move(got);
          double tpc = std::chrono::duration<double>(std::chrono::steady_clock::now() - tp0).count();
          std::lock_guard<std::mutex> lk(timeMu);
          tPc += tpc;
        }
        // the truth: its pool entry, else (answer outside the pool window or
        // removed) a generated candidate with the same stereo-blind key
        for (size_t i = 0; i < q.cand.size(); ++i) if (truthPool >= 0 && q.cand[i] == truthPool) q.truth = int(i);
        if (q.truth < 0 && !tkey.empty())
          for (size_t i = 0; i < q.cand.size(); ++i)
            if (q.cand[i] < 0 && q.gSmi[-q.cand[i] - 1] == tkey) { q.truth = int(i); break; }
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
  if (doGen || rmTruth) {
    size_t nG = 0, nRaw = 0, tRaw = 0, tKept = 0, tGenTruth = 0, qWithGen = 0;
    std::vector<int> per;
    for (auto& q : Q) {
      nG += q.gSmi.size(); nRaw += q.nGenRaw; tRaw += q.truthGenRaw; tKept += q.truthGenKept;
      per.push_back(int(q.gSmi.size())); qWithGen += !q.gSmi.empty();
      if (q.truth >= 0 && q.cand[q.truth] < 0) ++tGenTruth;
    }
    std::sort(per.begin(), per.end());
    std::printf("generation: %.1f thread-s (%.1f ms/query); distinct products %.1f/query; kept %.1f/query (median %d, "
                "queries with any %zu); truth among products %zu (%.3f), kept %zu (%.3f); truth = generated candidate in %zu; "
                "dropped: answer's stereo-blind duplicates in pool %ld, generated copies of an in-pool answer %ld, outside window %ld\n",
                tGen, 1000.0 * tGen / std::max<size_t>(1, Q.size()), double(nRaw) / Q.size(), double(nG) / Q.size(),
                per.empty() ? 0 : per[per.size() / 2], qWithGen, tRaw, double(tRaw) / Q.size(), tKept,
                double(tKept) / Q.size(), tGenTruth, long(nRmDup), long(nTruthDup), long(nOutWin));
    if (!detPath.empty()) {
      std::ofstream df(dir + "/" + detPath);
      for (auto& q : Q) df << q.det;
    }
  }

  if (doPc) {
    size_t nWin = 0, tIn = 0, tKept = 0, nFull = 0, tTop[5] = {0, 0, 0, 0, 0};
    std::vector<int> wins;
    for (auto& q : Q) {
      nWin += q.pcWin; wins.push_back(q.pcWin); tIn += q.pcTruthInWin; tKept += q.pcTruthKept; nFull += q.pc.size();
      if (q.pcTruthWin >= 0) {
        int r = q.pcTruthWin;
        tTop[0] += r < 25; tTop[1] += r < 100; tTop[2] += r < 300; tTop[3] += r < 1000; tTop[4] += 1;
      }
    }
    std::sort(wins.begin(), wins.end());
    std::printf("pubchem: %.1f thread-s (%.1f ms/query); window %.0f/query (median %d, max %d); truth in window %zu (%.3f)"
                " [rank by rev+pop: <25 %zu, <100 %zu, <300 %zu, <1000 %zu]; truth among fully scored %zu (%.3f); full %.1f/query\n",
                tPc, 1000.0 * tPc / std::max<size_t>(1, Q.size()), double(nWin) / Q.size(), wins.empty() ? 0 : wins[wins.size() / 2],
                wins.empty() ? 0 : wins.back(), tIn, double(tIn) / Q.size(), tTop[0], tTop[1], tTop[2], tTop[3], tKept,
                double(tKept) / Q.size(), double(nFull) / Q.size());
    if (opt.count("pcall")) {
      std::ofstream f(dir + "/" + opt["pcall"]);
      for (auto& q : Q) f << q.pcAll;
    }
  }

  // MRR@25 over all queries (truth absent = 0), with ties broken pessimistically
  auto evaluate = [&](const Params& P, std::vector<std::vector<float>>* scoresOut) {
    std::vector<double> rr(Q.size(), 0.0);
    if (scoresOut) scoresOut->assign(Q.size(), {});
    std::atomic<size_t> nx(0);
    std::vector<std::thread> tt;
    for (unsigned t = 0; t < nThr; ++t)
      tt.emplace_back([&]() {
        std::vector<std::pair<double, double>> pred;
        for (;;) {
          size_t qi = nx++;
          if (qi >= Q.size()) break;
          const Query& q = Q[qi];
          std::vector<float> sc(q.cand.size(), 0.f);
          float eps = std::exp(P.logEps);
          if (!keepFrags) sc = q.phys;  // eval: scored with the loaded params during enumeration
          for (size_t c = 0; c < q.cand.size() && keepFrags; ++c) {
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
  auto inLibOf = [&](int c) { return haveLib && c >= 0 && libSpecOf.count(pname[c]) > 0; };
  // generated candidates: G<k> (the truth's own name when it is the answer, so mrr.py scores the CSV)
  auto candName = [&](const Query& q, size_t i) -> std::string {
    int c = q.cand[i];
    if (c >= 0) return pname[c];
    if (int(i) == q.truth) return q.name;
    return "G" + std::to_string(-c - 1);
  };
  auto candSmiles = [&](const Query& q, size_t i) -> std::string {
    int c = q.cand[i];
    return c >= 0 ? psmi[c] : q.gSmi[-c - 1];
  };
  auto genTerm = [&](const Query& q, size_t i, const std::vector<double>& g) {
    double t = 0;
    for (int k = 0; k < NG; ++k) t += g[k] * q.gF[i][k];
    return float(t);
  };
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
           << int(inLibOf(q.cand[i])) << " " << (q.cand[i] >= 0 ? pname[q.cand[i]] : "GEN" + std::to_string(-q.cand[i] - 1))
           << " " << q.gF[i][0] << " " << q.gF[i][1] << " " << q.gF[i][2] << " " << q.gF[i][3] << " " << q.gF[i][4]
           << " " << q.gF[i][5] << "\n";
    }
    std::printf("wrote features %s\n", featPath.c_str());
  }
  std::printf("MRR@25 %.4f\n", m);
  if (keepFrags) {
    Params A = P; A.betaPos = 0.f; A.betaNeg = 0.f;
    std::printf("ablation: no charge retention %.4f\n", evaluate(A, nullptr));
  }
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
          f[i] += genTerm(Q[qi], i, gw);
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
    if (doGrid)
    for (float wr : {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 5.f}) {
      if (!haveRev && wr > 0.f) break;
      for (float w : {-1.f, 0.f, 0.25f, 0.5f, 1.f}) {
        if (!haveMlp && w != 0.f) continue;
        std::printf("fusion w=%-5g wr=%-5g MRR %.4f\n", w, wr, fused(w, wr, 0.f, nullptr));
      }
    }
    if (haveLib && doGrid) {
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
      // optional: NG generation-feature weights after the 8 base weights
      {
        std::vector<double> g(NG, 0.0);
        int k = 0;
        while (k < NG && (wf >> g[k])) ++k;
        if (k == NG) { gw = g; std::printf("learned generation weights read (%d)\n", k); }
      }
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
          float inLib = inLibOf(q.cand[i]) ? 1.f : 0.f;
          res[qi][i] = float(b[0] * zp[i] + b[1] * zm[i] + b[2] * zr[i] + b[3] * std::pow(l, 32.f) +
                             b[4] * std::pow(l, 8.f) + b[5] * l + b[6] * (haveLib ? q.anaP[i] : 0.f) +
                             b[7] * inLib) + genTerm(q, i, gw);
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
  // score-mode insertion compares PubChem picks (which have no FPNet score)
  // with the main list's scores BEFORE the FPNet term
  const std::vector<std::vector<float>> scoresNoFp = scores;
  if (!fpPath.empty() && fuseF != 0.f) {
    std::unordered_map<std::string, std::unordered_map<std::string, float>> fps;
    {
      std::ifstream f(dir + "/" + fpPath);
      if (!f) { std::fprintf(stderr, "cannot read %s\n", (dir + "/" + fpPath).c_str()); return 1; }
      std::string line;
      size_t nl = 0;
      while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        size_t t1 = line.find('\t'); if (t1 == std::string::npos) continue;
        size_t t2 = line.find('\t', t1 + 1); if (t2 == std::string::npos) continue;
        float v = float(std::atof(line.c_str() + t2 + 1));
        auto& m = fps[line.substr(0, t1)];
        auto ins = m.emplace(line.substr(t1 + 1, t2 - t1 - 1), v);
        if (!ins.second) ins.first->second = std::max(ins.first->second, v);
        ++nl;
      }
      std::printf("fpnet scores: %zu lines, %zu queries\n", nl, fps.size());
    }
    double tot = 0;
    size_t nHave = 0, nMiss = 0, nQ0 = 0;
    for (size_t qi = 0; qi < Q.size(); ++qi) {
      const Query& q = Q[qi];
      if (scores[qi].size() != q.cand.size()) continue;
      auto fq = fps.find(q.name);
      std::vector<float> v(q.cand.size());
      std::vector<char> have(q.cand.size(), 0);
      double m = 0; int k = 0;
      if (fq != fps.end())
        for (size_t i = 0; i < q.cand.size(); ++i) {
          if (q.cand[i] < 0) continue;  // generated: no FPNet score (query mean)
          auto c = fq->second.find(pname[q.cand[i]]);
          if (c != fq->second.end()) { v[i] = c->second; have[i] = 1; m += c->second; ++k; }
        }
      nHave += k; nMiss += q.cand.size() - k;
      if (k == 0) { ++nQ0; continue; }
      m /= k;
      for (size_t i = 0; i < v.size(); ++i) if (!have[i]) v[i] = float(m);
      auto zf = zscore(v);
      for (size_t i = 0; i < v.size(); ++i) scores[qi][i] += fuseF * zf[i];
      if (q.truth >= 0) {
        int better = 0;
        for (size_t i = 0; i < v.size(); ++i)
          if (int(i) != q.truth && scores[qi][i] >= scores[qi][q.truth]) ++better;
        if (better < 25) tot += 1.0 / (better + 1);
      }
    }
    std::printf("output + %g z(fpnet): candidates with score %zu, without %zu, queries without any %zu, MRR@25 %.4f\n",
                fuseF, nHave, nMiss, nQ0, tot / std::max<size_t>(1, Q.size()));
  }
  // final order per query (+ the PubChem slot insertion)
  std::vector<std::vector<std::string>> plain(Q.size()), inserted(Q.size());
  size_t nGated = 0;
  for (size_t qi = 0; qi < Q.size(); ++qi) {
    Query& q = Q[qi];
    std::vector<int> o(q.cand.size());
    for (size_t i = 0; i < o.size(); ++i) o[i] = int(i);
    std::sort(o.begin(), o.end(), [&](int a, int b) { return scores[qi][a] > scores[qi][b]; });
    if (scores[qi].size() == q.cand.size()) {
      q.top1 = o.size() > 0 ? scores[qi][o[0]] : 0.f;
      q.top2 = o.size() > 1 ? scores[qi][o[1]] : q.top1 - 1e3f;
    }
    for (size_t i = 0; i < o.size() && i < 25; ++i) {
      plain[qi].push_back(writeSmiles ? candSmiles(q, o[i]) : candName(q, o[i]));
      if (scoresNoFp[qi].size() == q.cand.size()) q.topS.push_back(scoresNoFp[qi][o[i]]);
    }
    if (!doPc) continue;
    const bool open = q.libMax < pcGate && (o.size() < 2 || q.top1 - q.top2 < pcGMarg);
    std::vector<std::string> pick;
    std::vector<float> pickS;  // score-mode: the pick's score on the main list's scale
    if (open) {
      ++nGated;
      for (size_t i = 0; i < q.pc.size() && int(pick.size()) < pcK; ++i) {
        pick.push_back(writeSmiles ? q.pc[i].smi : (q.pc[i].truth ? q.name : "P" + std::to_string(q.pc[i].cid)));
        pickS.push_back(q.pc[i].fused - pcPop * q.pc[i].pop + pcSW * q.pc[i].pop + pcBias);
      }
    }
    std::set<int> sl(pcSlots.begin(), pcSlots.end());
    size_t a = 0, b = 0;
    auto& L = inserted[qi];
    for (int pos = 1; pos <= 25; ++pos) {
      bool take;
      if (pcScoreMode)  // merge by score, never above slot pcMin
        take = b < pick.size() && pos >= pcMin && (a >= q.topS.size() || pickS[b] > q.topS[a]);
      else
        take = sl.count(pos) && b < pick.size();
      if (take) L.push_back(pick[b++]);
      else if (a < plain[qi].size()) L.push_back(plain[qi][a++]);
      else if (b < pick.size() && !pcScoreMode) L.push_back(pick[b++]);
      else break;
    }
  }
  auto writeCsv = [&](const std::string& path, const std::vector<std::vector<std::string>>& lists) {
    std::ofstream out(dir + "/" + path);
    out << "molecule_id,smiles\n";
    for (size_t qi = 0; qi < Q.size(); ++qi) {
      out << Q[qi].name << ",";
      for (size_t i = 0; i < lists[qi].size(); ++i) out << (i ? ";" : "") << lists[qi][i];
      out << "\n";
    }
  };
  // production (pc=1 without pcout): out.csv IS the inserted list
  if (!outPath.empty()) writeCsv(outPath, doPc && !opt.count("pcout") ? inserted : plain);
  if (doPc) {
    double mPlain = 0, mIns = 0;
    for (size_t qi = 0; qi < Q.size(); ++qi) {
      for (size_t i = 0; i < plain[qi].size(); ++i) if (plain[qi][i] == Q[qi].name) { mPlain += 1.0 / double(i + 1); break; }
      for (size_t i = 0; i < inserted[qi].size(); ++i) if (inserted[qi][i] == Q[qi].name) { mIns += 1.0 / double(i + 1); break; }
    }
    std::printf("pubchem slots: gate open for %zu / %zu queries; MRR@25 (eval names) plain %.4f -> inserted %.4f\n", nGated,
                Q.size(), mPlain / double(Q.size()), mIns / double(Q.size()));
    if (opt.count("pcout")) writeCsv(opt["pcout"], inserted);
    if (opt.count("pcdet")) {
      std::ofstream f(dir + "/" + opt["pcdet"]);
      for (auto& q : Q) {
        f << "Q\t" << q.name << "\t" << q.libMax << "\t" << q.top1 << "\t" << q.top2 << "\t" << q.anaTop << "\t"
          << q.cand.size() << "\t" << q.truth << "\t" << q.pcWin << "\t" << int(q.pcTruthInWin) << "\t" << q.pcTruthWin
          << "\t" << q.pc.size();
        for (float s : q.topS) f << "\t" << s;
        f << "\n";
        for (auto& c : q.pc)
          f << c.cid << "\t" << int(c.truth) << "\t" << c.rev << "\t" << c.pop << "\t" << c.phys << "\t" << c.mlp << "\t"
            << c.ana << "\t" << c.fused << "\t" << c.zp << "\t" << c.zm << "\t" << c.zr << "\t" << c.smi << "\n";
      }
    }
  }
  return 0;
}
