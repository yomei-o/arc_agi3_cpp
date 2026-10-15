// phys_qm = phys_rank + GFN2-xTB-derived formulas (2026-10-15, qm_STATUS.md):
//   QM_COEF=<coef.txt> (casmi/py/qm/qm_fit.py output)
//   QM_DIR=1  per-bond barrier E[class] + kappa*0.1*(dE_formula - median_class) and the
//             directional charge share + gDir*0.1*(dE_other - dE_self) (kcal/mol)
//   QM_PA=1   QM site proton-affinity table instead of sitePA()
//   QM_FIT=i,j  coordinate search over these parameters only (kappa 25, gDir 26, betaPos 9)
//   QM_GRID="k1,k2;g1,g2"  grid over (kappa, gDir) before the search
//   QM_TIES=avg  MRR with ties averaged over a random order (default: pessimistic, as phys_rank)
//   phys_qm dump <coef.txt> <mols.txt> <n>   per-bond formula values (checked against Python)
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

bool gTieAvg = false;  // QM_TIES=avg

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

// ---- QM-derived formulas (GFN2-xTB, fitted by qm_fit.py into qm_coef.txt) ----
//
// Directional even-electron cleavage energy of a protonated molecule,
//   MH+ -> A+ + B-H   (A keeps the charge; B leaves with an H on its cut atom b)
// relative to the most stable MH+, as a linear formula in NDIR graph
// descriptors of the cation site a and the leaving atom b (the same rules as
// qm_fit.py dfeat()). Proton affinity of a site: a table over NPAT site types
// + paLn ln(n_heavy/30) (+ paDon * o/p donors for an aromatic C-H).

constexpr int NDIR = 25, NPAT = 24;
struct QmCoef {
  bool have = false;
  float wDir[NDIR] = {}, wBond[NDIR] = {};  // directional (all rows) / bond level (QM-preferred direction rows)
  float med[NCLS] = {};
  float paT[NPAT] = {};
  float paLn = 0.f, paDon = 0.f, paMed = 0.f;
  bool usePA = false;   // QM_PA=1: QM site table replaces sitePA()
  bool useDir = false;  // QM_DIR=1: per-bond barrier and directional charge share from the formula
} QC;

bool loadQmCoef(const std::string& p) {
  std::ifstream f(p);
  if (!f) return false;
  std::string tag;
  while (f >> tag) {
    if (tag == "dir") for (float& x : QC.wDir) f >> x;
    else if (tag == "bond") for (float& x : QC.wBond) f >> x;
    else if (tag == "med") for (float& x : QC.med) f >> x;
    else if (tag == "pa") { for (float& x : QC.paT) f >> x; f >> QC.paLn >> QC.paDon >> QC.paMed; }
  }
  QC.have = true;
  return true;
}

bool hasDoubleTo(const RDKit::ROMol& m, const RDKit::Atom* a, int z) {
  for (auto b : m.atomBonds(a))
    if (b->getBondType() == RDKit::Bond::DOUBLE && b->getOtherAtom(a)->getAtomicNum() == z) return true;
  return false;
}

// paMax: the molecule's most basic site by the PA table (mobile proton: a strongly basic
// site holds the proton and raises every charge-directed cleavage)
int paType(const RDKit::ROMol& m, const RDKit::Atom* a);

float dirEnergy(const RDKit::ROMol& m, int ai, int bi, int nA, int nB, float paMax, const float* w = QC.wDir) {
  const RDKit::Atom* a = m.getAtomWithIdx(ai);
  const RDKit::Atom* b = m.getAtomWithIdx(bi);
  int za = a->getAtomicNum(), zb = b->getAtomicNum();
  int deg = 0, nO = 0, nN = 0, ar = 0, al = 0, co = 0;
  for (auto bd : m.atomBonds(a)) {
    const RDKit::Atom* o = bd->getOtherAtom(a);
    if (int(o->getIdx()) == bi || o->getAtomicNum() == 1) continue;
    ++deg;
    if (bd->getBondType() == RDKit::Bond::SINGLE && !o->getIsAromatic()) {
      if (o->getAtomicNum() == 8) ++nO;
      else if (o->getAtomicNum() == 7 && !hasDoubleO(m, o)) ++nN;
    }
    if (o->getIsAromatic()) ar = 1;
    else if (o->getAtomicNum() == 6)
      for (auto b2 : m.atomBonds(o))
        if (b2->getBondType() == RDKit::Bond::DOUBLE) {
          int z2 = b2->getOtherAtom(o)->getAtomicNum();
          if (z2 == 6) al = 1;
          else if (z2 == 8) co = 1;
        }
  }
  const int tb = paType(m, b);
  const float paB = tb >= 0 ? QC.paT[tb] : 0.f;  // PA-table value of the leaving atom
  bool bCO = zb == 6 && hasDoubleO(m, b);
  if (zb == 8)
    for (auto bd : m.atomBonds(b)) {
      const RDKit::Atom* o = bd->getOtherAtom(b);
      if (int(o->getIdx()) != ai && o->getAtomicNum() == 6 && hasDoubleO(m, o)) bCO = true;
    }
  const float f[NDIR] = {1.f, float(za == 7), float(za == 8), float(za == 16), float(a->getIsAromatic()),
                         float(za == 6 && hasDoubleO(m, a)), float(deg == 0), float(deg == 2), float(deg >= 3),
                         float(std::min(nO, 2)), float(std::min(nN, 1)), float(ar && !a->getIsAromatic()),
                         float(al), float(co), float(m.getRingInfo()->numAtomRings(ai) > 0),
                         float(hasDoubleTo(m, a, 6)), float(zb == 7), float(zb == 8), float(zb == 16),
                         float(b->getIsAromatic()), float(bCO), paB, paMax, std::log(float(std::max(1, nA))),
                         std::log(float(std::max(1, nB)))};
  float e = 0.f;
  for (int i = 0; i < NDIR; ++i) e += w[i] * f[i];
  return e;
}

// site type index as qm_fit.py pa_type(), -1 = not a site
enum PaT { N_PYRIDINE, N_PYRROLE, N_AMIDE, N_ANILINE, N_IMINE, N_AMINE1, N_AMINE2, N_AMINE3, N_OTHER,
           O_AMIDECO, O_ESTERCO, O_ACIDCO, O_KETONE, O_KETONECONJ, O_ALCOHOL, O_ETHER, O_RINGETHER, O_PHENOL,
           O_ARYLETHER, O_ESTERSINGLE, O_AROM, O_OTHER, S_ANY, C_AROM };

int paType(const RDKit::ROMol& m, const RDKit::Atom* a) {
  int z = a->getAtomicNum();
  if (z == 7) {
    if (a->getFormalCharge() != 0) {
      if (a->getFormalCharge() > 0 && a->getTotalNumHs() == 0) {
        bool negNb = false;
        for (auto bd : m.atomBonds(a)) if (bd->getOtherAtom(a)->getFormalCharge() < 0) negNb = true;
        if (!negNb) return -1;  // quaternary N+: no lone pair (qm2.py skips it)
      }
      return N_OTHER;
    }
    if (a->getIsAromatic()) return (a->getTotalNumHs() > 0 || a->getDegree() == 3) ? N_PYRROLE : N_PYRIDINE;
    bool amide = false, aryl = false, imine = false, sulf = false;
    int nC = 0;
    for (auto bd : m.atomBonds(a)) {
      const RDKit::Atom* o = bd->getOtherAtom(a);
      if (bd->getBondType() == RDKit::Bond::DOUBLE) imine = true;
      if (o->getAtomicNum() == 6) {
        ++nC;
        if (hasDoubleO(m, o)) amide = true;
        if (o->getIsAromatic()) aryl = true;
      }
      if (o->getAtomicNum() == 15 || o->getAtomicNum() == 16) sulf = true;
    }
    if (sulf) return N_OTHER;
    if (amide) return N_AMIDE;
    if (imine) return N_IMINE;
    if (aryl) return N_ANILINE;
    return nC >= 3 ? N_AMINE3 : nC == 2 ? N_AMINE2 : N_AMINE1;
  }
  if (z == 8) {
    if (a->getFormalCharge() != 0) return O_OTHER;
    if (a->getIsAromatic()) return O_AROM;
    for (auto bd : m.atomBonds(a))
      if (bd->getBondType() == RDKit::Bond::DOUBLE) {
        const RDKit::Atom* c = bd->getOtherAtom(a);
        if (c->getAtomicNum() != 6) return O_OTHER;
        bool nN = false, arN = false, cc = false;
        int oN = 0;  // 0 none, 1 ester-like, 2 acid
        for (auto b2 : m.atomBonds(c)) {
          const RDKit::Atom* x = b2->getOtherAtom(c);
          if (x == a) continue;
          if (x->getAtomicNum() == 7) nN = true;
          if (x->getAtomicNum() == 8) oN = x->getTotalNumHs() == 0 ? 1 : 2;
          if (x->getIsAromatic()) arN = true;
          if (x->getAtomicNum() == 6)
            for (auto b3 : m.atomBonds(x)) if (b3->getBondType() == RDKit::Bond::DOUBLE) cc = true;
        }
        if (nN) return O_AMIDECO;
        if (oN == 2) return O_ACIDCO;
        if (oN == 1) return O_ESTERCO;
        return (arN || cc) ? O_KETONECONJ : O_KETONE;
      }
    bool est = false, aro = false;
    for (auto bd : m.atomBonds(a)) {
      const RDKit::Atom* o = bd->getOtherAtom(a);
      if (o->getAtomicNum() != 6 && o->getAtomicNum() != 1) return O_OTHER;
      if (o->getAtomicNum() == 6 && hasDoubleO(m, o)) est = true;
      if (o->getIsAromatic()) aro = true;
    }
    if (est) return O_ESTERSINGLE;
    if (aro) return a->getTotalNumHs() > 0 ? O_PHENOL : O_ARYLETHER;
    if (a->getTotalNumHs() > 0) return O_ALCOHOL;
    return m.getRingInfo()->numAtomRings(a->getIdx()) > 0 ? O_RINGETHER : O_ETHER;
  }
  if (z == 16) return S_ANY;
  if (z == 6 && a->getIsAromatic() && a->getTotalNumHs() > 0) return C_AROM;
  return -1;
}

// aromatic C-H: exocyclic single-bonded O/N donors on aromatic atoms at graph distance 1 or 3
int aromDonors(const RDKit::ROMol& m, const RDKit::Atom* a) {
  const int n = int(m.getNumAtoms());
  std::vector<int> d(n, -1), q{int(a->getIdx())};
  d[a->getIdx()] = 0;
  for (size_t h = 0; h < q.size(); ++h) {
    int u = q[h];
    if (d[u] >= 3) continue;
    for (auto bd : m.atomBonds(m.getAtomWithIdx(u))) {
      int v = int(bd->getOtherAtom(m.getAtomWithIdx(u))->getIdx());
      if (d[v] < 0) { d[v] = d[u] + 1; q.push_back(v); }
    }
  }
  int c = 0;
  for (int i = 0; i < n; ++i) {
    if ((d[i] != 1 && d[i] != 3) || !m.getAtomWithIdx(i)->getIsAromatic()) continue;
    const RDKit::Atom* o = m.getAtomWithIdx(i);
    for (auto bd : m.atomBonds(o)) {
      const RDKit::Atom* x = bd->getOtherAtom(o);
      if (!x->getIsAromatic() && (x->getAtomicNum() == 7 || x->getAtomicNum() == 8) &&
          bd->getBondType() == RDKit::Bond::SINGLE) ++c;
    }
  }
  return c;
}

// QM site PA (kcal/mol, +150 so every site stays > 0 for softmax(); only
// differences matter). The size term is added per fragment in enumerate().
float sitePAqm(const RDKit::ROMol& m, const RDKit::Atom* a) {
  int t = paType(m, a);
  if (t < 0) return 0.f;
  return 150.f + QC.paT[t] + (t == C_AROM ? QC.paDon * float(aromDonors(m, a)) : 0.f);
}

float molPaMax(const RDKit::ROMol& m) {
  float mx = -1e9f;
  for (const auto a : m.atoms()) {
    int t = paType(m, a);
    if (t >= 0) mx = std::max(mx, QC.paT[t]);
  }
  return mx > -1e8f ? mx : QC.paMed;
}

// ---- per-candidate fragment table --------------------------------------

struct Frag {
  float mass;   // neutral sub-graph mass with its own hydrogens
  uint8_t cls;
  float pa, paOther;      // soft-max site PA of this fragment / its partner
  float acid, acidOther;  // same for acidity
  uint8_t cls2 = 255;     // second cut's class for a two-bond (sequential) cleavage
  float dq = 0.f;         // QM formula barrier deviation (kcal/mol, bond level minus class median; summed over cuts)
  float dSelf = 0.f, dOther = 0.f;  // directional cleavage energy with this fragment / its partner as the cation
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
    pa[i] = QC.usePA ? sitePAqm(mol, mol.getAtomWithIdx(i)) : sitePA(mol, mol.getAtomWithIdx(i));
    ac[i] = siteAcid(mol, mol.getAtomWithIdx(i));
  }
  // fragment PA size term (QM: larger ions bind the proton more strongly)
  auto paSize = [&](const std::vector<int>& idx) {
    return QC.usePA ? QC.paLn * std::log(std::max<size_t>(1, idx.size()) / 30.0f) : 0.f;
  };
  // per-bond QM barrier deviation and directional energies (single acyclic bonds)
  std::vector<float> bondDq(mol.getNumBonds(), 0.f);
  const float pmx = QC.useDir ? molPaMax(mol) : 0.f;
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
    for (size_t i = 0; i < frags.size(); ++i) {
      p[i] = score(frags[i], pa);
      if (p[i] > 0.f) p[i] += paSize(frags[i]);
      a[i] = score(frags[i], ac);
    }
    // QM: directional energies for a single-bond cut into two pieces
    std::vector<float> dir(frags.size(), 0.f);
    float dq = 0.f;
    if (QC.useDir) {
      for (unsigned b : bonds) dq += bondDq[b];
      const RDKit::Bond* bd = mol.getBondWithIdx(bonds[0]);
      if (bonds.size() == 1 && frags.size() == 2 && bd->getBondType() == RDKit::Bond::SINGLE) {
        int x = int(bd->getBeginAtomIdx()), y = int(bd->getEndAtomIdx());
        int fx = std::find(frags[0].begin(), frags[0].end(), x) != frags[0].end() ? 0 : 1;
        int nx = int(frags[fx].size()), ny = int(frags[1 - fx].size());
        dir[fx] = dirEnergy(mol, x, y, nx, ny, pmx);
        dir[1 - fx] = dirEnergy(mol, y, x, ny, nx, pmx);
      }
    }
    for (size_t i = 0; i < frags.size(); ++i) {
      // partner = best competing fragment of the same cut
      float po = 0.f, ao = 0.f;
      for (size_t j = 0; j < frags.size(); ++j)
        if (j != i) { po = std::max(po, p[j]); ao = std::max(ao, a[j]); }
      Frag fr{float(forward_model::fragmentMass(mol, frags[i])), uint8_t(cls), p[i], po, a[i], ao,
              uint8_t(cls2)};
      fr.dq = dq;
      if (frags.size() == 2) { fr.dSelf = dir[i]; fr.dOther = dir[1 - i]; }
      out.push_back(fr);
    }
  };
  const RDKit::RingInfo* ri0 = mol.getRingInfo();
  if (QC.useDir)
    for (const auto& b : mol.bonds()) {
      if (ri0->numBondRings(b->getIdx()) > 0 || b->getBondType() != RDKit::Bond::SINGLE) continue;
      auto fr = forward_model::fragmentsAfterCutting(mol, {unsigned(b->getIdx())});
      if (fr.size() != 2) continue;
      int x = int(b->getBeginAtomIdx()), y = int(b->getEndAtomIdx());
      int fx = std::find(fr[0].begin(), fr[0].end(), x) != fr[0].end() ? 0 : 1;
      int nx = int(fr[fx].size()), ny = int(fr[1 - fx].size());
      // direction chosen by the directional formula, energy from the bond-level one
      bool xKeeps = dirEnergy(mol, x, y, nx, ny, pmx) <= dirEnergy(mol, y, x, ny, nx, pmx);
      float e = xKeeps ? dirEnergy(mol, x, y, nx, ny, pmx, QC.wBond) : dirEnergy(mol, y, x, ny, nx, pmx, QC.wBond);
      bondDq[b->getIdx()] = e - QC.med[bondClass(mol, b)];
    }
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
  // QM terms: barrier + kappa * dq / 10 kcal; charge share + gDir * (dOther - dSelf) / 10 kcal
  float kappa = 0.0f, gDir = 0.0f;

  // file order keeps older files loadable: the original classes, betas,
  // k weights, eps, E2, then E[CGLYC] and dofSlope (added later)
  static constexpr int N = (NCLS - 1) + 2 + 5 + 5 + 1 + 1 + 1 + 1 + 2;
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
    if (i == 3) return &dofSlope;
    return i == 4 ? &kappa : &gDir;
  }
  void save(const std::string& p) {
    std::ofstream f(p);
    for (int i = 0; i < N; ++i) f << *at(i) << "\n";
  }
  bool load(const std::string& p) {
    std::ifstream f(p);
    // files written before E2 existed hold N-1 values; E2 keeps its default
    for (int i = 0; i < N; ++i) if (!(f >> *at(i))) return i >= N - 5;
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
      if (!neg && f.dSelf != f.dOther) d += P.gDir * 0.1 * (f.dOther - f.dSelf);
      if (metal) d = 0.0;  // a metal cation does not follow proton affinity
      share = 1.0 / (1.0 + std::exp(-d));
    }
    double e = P.E[f.cls] + (f.cls2 != 255 ? P.E[f.cls2] + P.E2 : 0.f) + (f.cls != INTACT ? dof : 0.0) +
               P.kappa * 0.1 * f.dq;
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
};

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
  if (argc < 7 && !(argc >= 5 && std::string(argv[1]) == "dump")) {
    std::fprintf(stderr, "usage: phys_rank fit|eval <dir> <qry> <pool_stem> <ppm> <params> [out.csv]\n");
    return 1;
  }
  if (std::string(argv[1]) == "dump") {
    // phys_qm dump <coef.txt> <mols.txt> <n>: per-bond formula values, to check against qm_fit.py
    if (!loadQmCoef(argv[2])) return 1;
    std::ifstream f(argv[3]);
    std::string line;
    int n = std::atoi(argv[4]);
    while (n-- > 0 && std::getline(f, line)) {
      if (!line.empty() && line.back() == '\r') line.pop_back();
      std::string name = line.substr(0, line.find('\t')), smi = line.substr(line.find('\t') + 1);
      std::unique_ptr<RDKit::RWMol> m(RDKit::SmilesToMol(smi));
      if (!m) continue;
      float pmx = molPaMax(*m);
      for (const auto a : m->atoms()) {
        int t = paType(*m, a);
        if (t >= 0) std::printf("S %s %u %d %.3f\n", name.c_str(), a->getIdx(), t, sitePAqm(*m, a) - 150.f);
      }
      for (const auto b : m->bonds()) {
        if (m->getRingInfo()->numBondRings(b->getIdx()) || b->getBondType() != RDKit::Bond::SINGLE) continue;
        auto fr = forward_model::fragmentsAfterCutting(*m, {unsigned(b->getIdx())});
        if (fr.size() != 2) continue;
        int x = int(b->getBeginAtomIdx()), y = int(b->getEndAtomIdx());
        int fx = std::find(fr[0].begin(), fr[0].end(), x) != fr[0].end() ? 0 : 1;
        int nx = int(fr[fx].size()), ny = int(fr[1 - fx].size());
        std::printf("B %s %u %d %d %.3f %.3f %.3f %.3f\n", name.c_str(), b->getIdx(), x, y,
                    dirEnergy(*m, x, y, nx, ny, pmx), dirEnergy(*m, y, x, ny, nx, pmx),
                    dirEnergy(*m, x, y, nx, ny, pmx, QC.wBond), dirEnergy(*m, y, x, ny, nx, pmx, QC.wBond));
      }
    }
    return 0;
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
  bool haveLib = !libStem.empty();
  // QM formulas: QM_COEF=<qm_coef.txt> QM_PA=1 (site table) QM_DIR=1 (per-bond barrier + directional share)
  if (const char* qc = std::getenv("QM_COEF")) {
    if (!loadQmCoef(qc)) { std::fprintf(stderr, "cannot read %s\n", qc); return 1; }
    const char* e1 = std::getenv("QM_PA");
    const char* e2 = std::getenv("QM_DIR");
    QC.usePA = e1 && std::atoi(e1);
    QC.useDir = e2 && std::atoi(e2);
    std::printf("QM coefficients %s: PA table %d, directional/barrier %d\n", qc, int(QC.usePA), int(QC.useDir));
  }
  if (const char* tv = std::getenv("QM_TIES")) gTieAvg = std::string(tv) == "avg";
  if (gTieAvg) std::printf("MRR ties: averaged over random order\n");
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
  std::atomic<size_t> next(0);
  std::vector<std::thread> th;
  for (unsigned t = 0; t < std::max(1u, std::thread::hardware_concurrency()); ++t)
    th.emplace_back([&]() {
      std::vector<float> acc(haveLib ? lhead.size() : 0, 0.f);
      std::vector<int32_t> touched;
      for (;;) {
        size_t qi = next++;
        if (qi >= Q.size()) break;
        Query& q = Q[qi];
        // Spectral analogs: library molecules whose spectra look like this
        // query's (any structure, precursor within 200 Da), top 80 by
        // similarity, with their fingerprints.
        std::vector<std::pair<float, std::vector<uint16_t>>> analogs;
        if (haveLib) {
          std::unordered_map<int, float> molBest;
          for (int s : q.spectra) {
            const Head& h = head[s];
            double nrm = 0;
            for (int i = 0; i < h.n; ++i) nrm += std::max(0.f, it[h.off + i]);
            nrm = std::sqrt(nrm) + 1e-12;
            touched.clear();
            for (int i = 0; i < h.n; ++i) {
              float wq = float(std::sqrt(std::max(0.f, it[h.off + i])) / nrm);
              int b = int(mz[h.off + i] * 100.f);
              // the peak's own bin and its nearer neighbour: +-0.005..0.01 Da
              int b2 = (mz[h.off + i] * 100.f - float(b)) < 0.5f ? b - 1 : b + 1;
              for (int bb : {b, b2}) {
                if (bb < 0 || bb >= NBIN) continue;
                for (int64_t k = binStart[bb]; k < binStart[bb + 1]; ++k) {
                  int32_t ls = postSpec[k];
                  if (acc[ls] == 0.f) touched.push_back(ls);
                  acc[ls] += wq * postW[k];
                }
              }
            }
            for (int32_t ls : touched) {
              float sim = std::min(1.f, acc[ls]);
              acc[ls] = 0.f;
              if (std::fabs(lhead[ls].precursor - h.precursor) > 200.f || sim < 0.3f) continue;
              auto& mb = molBest[lhead[ls].mol];
              mb = std::max(mb, sim);
            }
          }
          std::vector<std::pair<float, int>> top;
          for (auto& kv : molBest) top.emplace_back(kv.second, kv.first);
          std::sort(top.begin(), top.end(), [](auto& a, auto& b) { return a.first > b.first; });
          if (top.size() > 80) top.resize(80);
          for (auto& tp : top) {
            std::unique_ptr<RDKit::RWMol> am;
            try { am.reset(RDKit::SmilesToMol(lsmiles[tp.second])); } catch (...) { continue; }
            if (!am) continue;
            analogs.emplace_back(tp.first, ecfp::compute(*am, 2, 2048));
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
            float ap = 0.f;
            for (auto& an : analogs) {
              size_t i = 0, j = 0, inter = 0;
              while (i < cb.size() && j < an.second.size()) {
                if (cb[i] == an.second[j]) { ++inter; ++i; ++j; }
                else if (cb[i] < an.second[j]) ++i;
                else ++j;
              }
              size_t uni = cb.size() + an.second.size() - inter;
              float tan = uni ? float(inter) / float(uni) : 0.f;
              ap = std::max(ap, tan * an.first * an.first * an.first);
            }
            q.anaP.push_back(ap);
          }
        }
        auto tp = poolIdx.find(q.name);
        if (tp != poolIdx.end())
          for (size_t i = 0; i < q.cand.size(); ++i) if (q.cand[i] == tp->second) q.truth = int(i);
        if (qi % 100 == 0) std::printf("  enumerated %zu / %zu\n", qi, Q.size());
      }
    });
  for (auto& t : th) t.join();
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
            // ties: pessimistic (all tied candidates ahead, the historical metric) or, with
            // QM_TIES=avg, the expected reciprocal rank over a random order of the tied ones
            int better = 0, tied = 0;
            for (size_t c = 0; c < sc.size(); ++c) {
              if (int(c) == q.truth) continue;
              if (sc[c] > sc[q.truth]) ++better;
              else if (sc[c] == sc[q.truth]) ++tied;
            }
            if (!gTieAvg) {
              if (better + tied < 25) rr[qi] = 1.0 / (better + tied + 1);
            } else {
              double a = 0;
              for (int k = 0; k <= tied; ++k) if (better + k < 25) a += 1.0 / (better + k + 1);
              rr[qi] = a / (tied + 1);
            }
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
    // QM_FIT=i,j,...: coordinate search over these parameter indices only
    // (kappa = N-2, gDir = N-1, betaPos = NCLS-1)
    std::set<int> fitSet;
    if (const char* fs = std::getenv("QM_FIT")) {
      std::string t(fs);
      size_t p0 = 0;
      while (p0 < t.size()) {
        size_t p1 = t.find(',', p0);
        if (p1 == std::string::npos) p1 = t.size();
        fitSet.insert(std::atoi(t.substr(p0, p1 - p0).c_str()));
        p0 = p1 + 1;
      }
    }
    if (const char* ip = std::getenv("QM_INIT")) {
      if (P.load(dir + "/" + ip)) std::printf("init from %s\n", ip);
    }
    double best = evaluate(P, nullptr);
    std::printf("initial MRR %.4f\n", best);
    // QM_GRID="k1,k2,..;g1,g2,..": exhaustive grid over (kappa, gDir) first
    if (const char* gs = std::getenv("QM_GRID")) {
      auto parse = [](const std::string& t) {
        std::vector<float> v;
        size_t p0 = 0;
        while (p0 < t.size()) {
          size_t p1 = t.find(',', p0);
          if (p1 == std::string::npos) p1 = t.size();
          v.push_back(float(std::atof(t.substr(p0, p1 - p0).c_str())));
          p0 = p1 + 1;
        }
        return v;
      };
      std::string g(gs);
      size_t sc = g.find(';');
      std::vector<float> ks = parse(g.substr(0, sc)), gd = parse(sc == std::string::npos ? "0" : g.substr(sc + 1));
      Params B = P;
      for (float k : ks)
        for (float d : gd) {
          Params T = P;
          T.kappa = k;
          T.gDir = d;
          double v = evaluate(T, nullptr);
          std::printf("  grid kappa %.4f gDir %.4f MRR %.4f\n", k, d, v);
          if (v > best + 1e-5) { best = v; B = T; }
        }
      P = B;
      std::printf("grid best kappa %.4f gDir %.4f MRR %.4f\n", P.kappa, P.gDir, best);
      std::fflush(stdout);
    }
    // coordinate search: each parameter tried at +-step, step halves when a
    // full sweep finds nothing; E[INTACT] and the k weights included.
    float step = 1.0f;
    for (int sweep = 0; sweep < 12 && step > 0.05f; ++sweep) {
      bool improved = false;
      for (int i = 0; i < Params::N; ++i) {
        if (!fitSet.empty() && !fitSet.count(i)) continue;
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
    std::printf(" E2=%.3f dofSlope=%.3f betaPos=%.3f kappa=%.3f gDir=%.3f\n", P.E2, P.dofSlope, P.betaPos, P.kappa, P.gDir);
    A = P; A.kappa = 0.f;
    std::printf("ablation: kappa 0 %.4f\n", evaluate(A, nullptr));
    A = P; A.gDir = 0.f;
    std::printf("ablation: gDir 0 %.4f\n", evaluate(A, nullptr));
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
  std::printf("MRR@25 %.4f  (kappa %.3f gDir %.3f betaPos %.3f)\n", m, P.kappa, P.gDir, P.betaPos);
  {
    Params A = P; A.kappa = 0.f; A.gDir = 0.f;
    std::printf("ablation: kappa=gDir=0 %.4f\n", evaluate(A, nullptr));
  }
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
    }
    libPow = fuseLPow;
    anaW = fuseA;
    if (fuseW > 0.f || fuseR > 0.f || fuseL > 0.f || fuseA > 0.f) {
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
