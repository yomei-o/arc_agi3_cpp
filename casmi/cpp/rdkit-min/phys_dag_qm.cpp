// phys_dag + GFN2-xTB-derived formulas (QM_COEF / QM_PA / QM_DIR, see phys_qm.cpp).
// Physics-shaped fragment intensities over a fragment DAG (absorbing Markov
// chain) instead of phys_rank's single cuts + ad-hoc labile-pair cuts.
// Same parameters, CLI, fit and fusion options as phys_rank; see the DAG
// section below for the model. Options from the environment:
//   DAG_DEPTH (2)  DAG_MAXNODES (2000)  DAG_LEAF (0: depth-cap exits lost,
//   1: depth-cap nodes observe everything)  DAG_THREADS (all cores)
//
//   phys_dag fit  <dir> <qry> <pool_stem> <ppm> <params_out> [params_init] [instrument]
//   phys_dag eval <dir> <qry> <pool_stem> <ppm> <params_in> [out.csv] [mlp.bin] [w] [smiles]
//                 [rev.bin] [wr] [lib_stem] [wl] [lib_pow] [wa] [features.txt]
#include <array>
#include <bit>
#include <chrono>
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
// DAG-only event classes: a ring 2-bond opening whose two bonds are 1 or 2
// apart around the ring (CFM-ID 4.0's ring-break distance); barrier
// E[RING or RINGAR] + dRing[0 or 1]. Distance >= 3 stays RING / RINGAR.
enum ClsX { RING1 = NCLS, RING2, RINGAR1, RINGAR2, NCLSX };

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
  float dRing[2] = {0.f, 0.f};  // ring-opening barrier offset, cut bonds 1 / 2 apart
  // "Voltage drop" (circuit analogy: a bond is a diode, an ion's flow a
  // current): every cleavage consumes internal energy softplus(E + vC), so a
  // node carries U = the flow-weighted energy already spent on the path from
  // the precursor, and its exits are slowed by exp(-vLambda * U). E2 * depth
  // is the special case "every step spends the same energy". vLambda = 0
  // switches it off.
  float vLambda = 0.0f, vC = 1.0f;
  // QM: barrier + kappa * 0.1 * dq (kcal); share + gDir * 0.1 * (dOther - dSelf) (kcal)
  float kappa = 0.0f, gDir = 0.0f;

  // file order keeps older files loadable: the original classes, betas,
  // k weights, eps, E2, then E[CGLYC] and dofSlope (added later)
  static constexpr int N = (NCLS - 1) + 2 + 5 + 5 + 1 + 1 + 1 + 1 + 2 + 2 + 2;
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
    if (i < 6) return &dRing[i - 4];
    if (i == 6) return &vLambda;
    if (i == 7) return &vC;
    return i == 8 ? &kappa : &gDir;
  }
  void save(const std::string& p) {
    std::ofstream f(p);
    for (int i = 0; i < N; ++i) f << *at(i) << "\n";
  }
  bool load(const std::string& p) {
    std::ifstream f(p);
    // files written before E2 existed hold N-1 values; E2 keeps its default
    for (int i = 0; i < N; ++i) if (!(f >> *at(i))) return i >= N - 9;
    return true;
  }
};

// ---- fragment DAG ---------------------------------------------------------
//
// Nodes = distinct connected atom subsets (dedupe by atom bitset), root = the
// intact molecule. An edge = one cleavage event of a node: a bridge of the
// node's induced subgraph, or two bonds of one SSSR ring lying wholly inside
// the node (when that disconnects it). Each event splits the node into two
// pieces; pieces with >= 2 heavy atoms are child nodes.
//
// Absorbing Markov chain over the DAG: from node n with exit rates
// k_e = exp(-E[cls(e)] - dof - E2 * depth(n)), K_n = sum_e k_e and the
// "observe" weight s = exp(-E[INTACT]),
//   P(observe n)       = s   / (s + K_n)
//   P(leave n via e)   = k_e / (s + K_n)
//   flow(root) = 1,  flow(c) = sum_{e: n->c} flow(n) P(e) share(c | e)
//   observed(n) = flow(n) P(observe n)
// share(c | e) = logistic(beta (S(c) - S(other piece))) (Cooks' kinetic
// method, as in phys_rank); 0.5 for metal adducts. This is the root row of
// the fundamental matrix (I-Q)^-1 times the absorption vector, computed by
// one pass in topological order (pieces are strict subsets, so decreasing
// atom count is a topological order). Flow into pieces that are not nodes
// (< 2 heavy atoms, node cap reached) and out of depth-cap nodes is lost
// (unpredicted), so the predicted spectrum is normalized by the total flow 1,
// not by the sum of the predicted peaks.
//
// For speed the DAG is pruned per query: only nodes whose m/z (any H shift)
// matches some peak of the query's spectra, and their ancestors, are kept;
// K_n is kept as a per-class count of all of n's events, so pruning does not
// change any flow.

struct DagOpts {
  int depth = 2;          // nodes deeper than this are not created
  int maxNodes = 2000;    // per molecule
  bool leafObserve = false;  // depth-cap nodes: observe everything (K=0) instead of losing their exits
};

// share-score differences in 0.01 kcal/mol (int16: memory at depth 3)
struct DagEdge { uint16_t child; uint8_t cls; int16_t dpos, dneg; int16_t dq = 0, ddir = 0; };  // dq, ddir in 0.1 kcal
inline int16_t q16(float x) { return int16_t(std::lround(std::max(-327.f, std::min(327.f, x)) * 100.f)); }
struct Match { uint16_t peak; uint16_t node; uint8_t kind; };  // kind 0..4 = H shift -2..2, 5 = metal

struct Dag {
  std::vector<uint8_t> depth;            // per kept node, topological order, root first
  std::vector<uint16_t> cnt;             // NCLSX per node: cleavage events by class (all of them)
  std::vector<float> s1, s2;             // NCLSX per node: sum of QM dq and dq^2 (kcal) over those events
  std::vector<uint32_t> eStart;          // CSR into edges
  std::vector<DagEdge> edges;
  std::vector<std::vector<Match>> match; // per query spectrum
};

struct DagStats {
  std::atomic<long long> nodes{0}, kept{0}, edges{0}, matches{0}, mols{0}, capped{0};
};

struct MolG {
  int n = 0, W = 0;
  std::vector<std::vector<std::pair<int, int>>> adj;  // (neighbour, bond)
  std::vector<int> bA, bB;
  std::vector<uint8_t> bCls, bArom;
  std::vector<std::vector<int>> rAtoms, rBonds;
  std::vector<double> am;  // atom + its hydrogens
  std::vector<float> pa, ac;
  std::vector<float> dA0, dB0, bA0, bB0;  // QM directional / bond-level energy without size terms, bA / bB side keeps the charge
  std::vector<uint8_t> heavy;
};

inline bool tbit(const uint64_t* b, int i) { return (b[i >> 6] >> (i & 63)) & 1; }
inline void sbit(uint64_t* b, int i) { b[i >> 6] |= uint64_t(1) << (i & 63); }

struct DagBuilder {
  const MolG& g;
  // scratch
  std::vector<int> disc, low, stack;
  std::vector<int> bridgesOut;
  int timer = 0;
  explicit DagBuilder(const MolG& gg) : g(gg), disc(gg.n, -1), low(gg.n, 0) {}

  void dfs(const uint64_t* bits, int u, int pb) {
    disc[u] = low[u] = timer++;
    for (auto [v, b] : g.adj[u]) {
      if (b == pb || !tbit(bits, v)) continue;
      if (disc[v] < 0) {
        dfs(bits, v, b);
        low[u] = std::min(low[u], low[v]);
        if (low[v] > disc[u]) bridgesOut.push_back(b);
      } else {
        low[u] = std::min(low[u], disc[v]);
      }
    }
  }
  void bridges(const uint64_t* bits) {
    bridgesOut.clear();
    timer = 0;
    int first = -1;
    for (int i = 0; i < g.n; ++i) if (tbit(bits, i)) { first = i; break; }
    if (first < 0) return;
    dfs(bits, first, -1);
    for (int i = 0; i < g.n; ++i) if (tbit(bits, i)) disc[i] = -1;
  }
  // component of `start` inside bits, without bonds x1/x2
  void comp(const uint64_t* bits, int start, int x1, int x2, uint64_t* out) {
    std::fill(out, out + g.W, 0);
    stack.clear();
    stack.push_back(start);
    sbit(out, start);
    while (!stack.empty()) {
      int u = stack.back(); stack.pop_back();
      for (auto [v, b] : g.adj[u]) {
        if (b == x1 || b == x2 || !tbit(bits, v) || tbit(out, v)) continue;
        sbit(out, v);
        stack.push_back(v);
      }
    }
  }
  // all cleavage events of the node: f(cls, A, B)
  template <class F> void cuts(const uint64_t* bits, F&& f) {
    std::vector<uint64_t> A(g.W), B(g.W);
    bridges(bits);
    std::vector<int> br = bridgesOut;
    for (int b : br) {
      comp(bits, g.bA[b], b, -1, A.data());
      for (int w = 0; w < g.W; ++w) B[w] = bits[w] & ~A[w];
      f(int(g.bCls[b]), A.data(), B.data(), b);
    }
    for (size_t r = 0; r < g.rAtoms.size(); ++r) {
      bool in = true;
      for (int a : g.rAtoms[r]) if (!tbit(bits, a)) { in = false; break; }
      if (!in) continue;
      const auto& rb = g.rBonds[r];
      for (size_t i = 0; i < rb.size(); ++i)
        for (size_t j = i + 1; j < rb.size(); ++j) {
          int b1 = rb[i], b2 = rb[j];
          comp(bits, g.bA[b1], b1, b2, A.data());
          if (tbit(A.data(), g.bB[b1])) continue;  // still connected: not a cleavage
          for (int w = 0; w < g.W; ++w) B[w] = bits[w] & ~A[w];
          // RingInfo lists a ring's bonds in ring order: distance around the ring
          int dist = int(std::min(j - i, rb.size() - (j - i)));
          bool ar = g.bArom[b1] || g.bArom[b2];
          int cls = dist == 1 ? (ar ? RINGAR1 : RING1) : dist == 2 ? (ar ? RINGAR2 : RING2) : (ar ? RINGAR : RING);
          f(cls, A.data(), B.data(), -1);
        }
    }
  }
  int heavyCount(const uint64_t* bits) const {
    int c = 0;
    for (int i = 0; i < g.n; ++i) if (tbit(bits, i) && g.heavy[i]) ++c;
    return c;
  }
  float soft(const uint64_t* bits, const std::vector<float>& v) const {
    float mx = 0.f;
    for (int i = 0; i < g.n; ++i) if (tbit(bits, i)) mx = std::max(mx, v[i]);
    if (mx <= 0.f) return 0.f;
    double s = 0;
    for (int i = 0; i < g.n; ++i) if (tbit(bits, i) && v[i] > 0.f) s += std::exp((v[i] - mx) / 3.0);
    return mx + 3.0f * float(std::log(s));
  }
};

MolG molGraph(const RDKit::ROMol& mol) {
  MolG g;
  g.n = int(mol.getNumAtoms());
  g.W = (g.n + 63) / 64;
  g.adj.resize(g.n);
  const RDKit::PeriodicTable* pt = RDKit::PeriodicTable::getTable();
  g.am.resize(g.n); g.pa.resize(g.n); g.ac.resize(g.n); g.heavy.resize(g.n);
  for (int i = 0; i < g.n; ++i) {
    const RDKit::Atom* a = mol.getAtomWithIdx(i);
    g.am[i] = pt->getMostCommonIsotopeMass(a->getAtomicNum()) +
              double(a->getTotalNumHs(false)) * pt->getMostCommonIsotopeMass(1);
    g.pa[i] = QC.usePA ? sitePAqm(mol, a) : sitePA(mol, a);
    g.ac[i] = siteAcid(mol, a);
    g.heavy[i] = a->getAtomicNum() > 1;
  }
  const float pmx = QC.useDir ? molPaMax(mol) : 0.f;
  for (const auto& b : mol.bonds()) {
    int i = int(b->getIdx());
    if (int(g.bA.size()) <= i) { g.bA.resize(i + 1); g.bB.resize(i + 1); g.bCls.resize(i + 1); g.bArom.resize(i + 1); }
    g.bA[i] = int(b->getBeginAtomIdx());
    g.bB[i] = int(b->getEndAtomIdx());
    g.bCls[i] = uint8_t(bondClass(mol, b));
    g.bArom[i] = b->getIsAromatic();
    if (QC.useDir) {
      if (int(g.dA0.size()) <= i) { g.dA0.resize(i + 1); g.dB0.resize(i + 1); g.bA0.resize(i + 1); g.bB0.resize(i + 1); }
      if (b->getBondType() == RDKit::Bond::SINGLE) {
        int x = g.bA[i], y = g.bB[i];
        g.dA0[i] = dirEnergy(mol, x, y, 1, 1, pmx);
        g.dB0[i] = dirEnergy(mol, y, x, 1, 1, pmx);
        g.bA0[i] = dirEnergy(mol, x, y, 1, 1, pmx, QC.wBond);
        g.bB0[i] = dirEnergy(mol, y, x, 1, 1, pmx, QC.wBond);
      } else {
        g.dA0[i] = g.dB0[i] = g.bA0[i] = g.bB0[i] = std::nanf("");
      }
    }
    g.adj[g.bA[i]].emplace_back(g.bB[i], i);
    g.adj[g.bB[i]].emplace_back(g.bA[i], i);
  }
  const RDKit::RingInfo* ri = mol.getRingInfo();
  for (unsigned r = 0; r < ri->numRings(); ++r) {
    g.rAtoms.push_back(ri->atomRings()[r]);
    g.rBonds.push_back(ri->bondRings()[r]);
  }
  return g;
}

constexpr double MASS_H = 1.00782503207, ELECTRON = 0.00054858;

// One query spectrum as the DAG builder needs it: peaks sorted by m/z.
struct SpecView {
  std::vector<std::pair<float, int>> sorted;
  int adduct;
};

Dag buildDag(const RDKit::ROMol& mol, const std::vector<SpecView>& specs, const DagOpts& O, DagStats& st,
             float tol) {
  MolG g = molGraph(mol);
  DagBuilder B(g);
  const int W = g.W;
  std::vector<uint64_t> bits;
  std::vector<uint8_t> depth;
  std::vector<int> heavyN;
  std::vector<float> mass;
  std::vector<std::vector<DagEdge>> tmpE;
  std::vector<std::array<uint16_t, NCLSX>> cnt;
  std::vector<std::array<float, NCLSX>> s1, s2;
  // QM energies of one bridge event (A holds g.bA[b]); dq = bond level - class median
  auto qmEvent = [&](int b, int cls, int hA, int hB, float& dq, float& eA, float& eB) {
    dq = eA = eB = 0.f;
    if (!QC.useDir || b < 0 || std::isnan(g.dA0[b])) return;
    const float la = std::log(float(std::max(1, hA))), lb = std::log(float(std::max(1, hB)));
    eA = g.dA0[b] + QC.wDir[NDIR - 2] * la + QC.wDir[NDIR - 1] * lb;
    eB = g.dB0[b] + QC.wDir[NDIR - 2] * lb + QC.wDir[NDIR - 1] * la;
    float e = eA <= eB ? g.bA0[b] + QC.wBond[NDIR - 2] * la + QC.wBond[NDIR - 1] * lb
                       : g.bB0[b] + QC.wBond[NDIR - 2] * lb + QC.wBond[NDIR - 1] * la;
    dq = e - QC.med[cls < NCLS ? cls : 0];
  };
  auto q10 = [](float x) { return int16_t(std::lround(std::max(-3000.f, std::min(3000.f, x)) * 10.f)); };
  std::unordered_map<std::string, int> index;
  auto key = [&](const uint64_t* b) { return std::string(reinterpret_cast<const char*>(b), size_t(W) * 8); };
  auto create = [&](const uint64_t* b, int d, int hv) {
    int id = int(depth.size());
    bits.insert(bits.end(), b, b + W);
    depth.push_back(uint8_t(d));
    heavyN.push_back(hv);
    double m = 0;
    for (int i = 0; i < g.n; ++i) if (tbit(b, i)) m += g.am[i];
    mass.push_back(float(m));
    tmpE.emplace_back();
    cnt.push_back({});
    s1.push_back({});
    s2.push_back({});
    index.emplace(key(b), id);
    return id;
  };
  {
    std::vector<uint64_t> all(W, 0);
    for (int i = 0; i < g.n; ++i) sbit(all.data(), i);
    create(all.data(), 0, B.heavyCount(all.data()));
  }
  bool capped = false;
  std::vector<uint64_t> cur(W);
  for (size_t u = 0; u < depth.size(); ++u) {
    if (depth[u] >= O.depth) continue;  // leaf: counted later, only if kept
    std::copy(bits.begin() + u * W, bits.begin() + (u + 1) * W, cur.begin());
    int du = depth[u];
    B.cuts(cur.data(), [&](int cls, const uint64_t* A, const uint64_t* Bb, int bnd) {
      ++cnt[u][cls];
      int hA = B.heavyCount(A), hB = B.heavyCount(Bb);
      float qdq, qeA, qeB;
      qmEvent(bnd, cls, hA, hB, qdq, qeA, qeB);
      s1[u][cls] += qdq;
      s2[u][cls] += qdq * qdq;
      int idA = -1, idB = -1;
      for (int side = 0; side < 2; ++side) {
        const uint64_t* P = side ? Bb : A;
        int hv = side ? hB : hA;
        if (hv < 2) continue;
        auto itn = index.find(key(P));
        int id = -1;
        if (itn != index.end()) id = itn->second;
        else if (int(depth.size()) < O.maxNodes) id = create(P, du + 1, hv);
        else capped = true;
        (side ? idB : idA) = id;
      }
      if (idA < 0 && idB < 0) return;
      float pA = B.soft(A, g.pa), pB = B.soft(Bb, g.pa);
      if (QC.usePA) {  // fragment size term of the QM PA formula
        if (pA > 0.f) pA += QC.paLn * std::log(float(std::max(1, hA)) / 30.f);
        if (pB > 0.f) pB += QC.paLn * std::log(float(std::max(1, hB)) / 30.f);
      }
      float aA = B.soft(A, g.ac), aB = B.soft(Bb, g.ac);
      if (idA >= 0) tmpE[u].push_back({uint16_t(idA), uint8_t(cls), q16(pA - pB), q16(aA - aB), q10(qdq), q10(qeB - qeA)});
      if (idB >= 0) tmpE[u].push_back({uint16_t(idB), uint8_t(cls), q16(pB - pA), q16(aB - aA), q10(qdq), q10(qeA - qeB)});
    });
  }
  const int N = int(depth.size());
  // match node masses against the query's peaks
  std::vector<std::vector<Match>> match(specs.size());
  std::vector<uint8_t> keep(N, 0);
  for (size_t s = 0; s < specs.size(); ++s) {
    int a = specs[s].adduct;
    bool neg = a >= 0 && a < 10 && ADDUCT_NEG[a];
    bool metal = a >= 0 && a < 10 && ADDUCT_METAL[a];
    double z = neg ? -1.0 : 1.0;
    const auto& sp = specs[s].sorted;
    for (int u = 0; u < N; ++u)
      for (int k = 0; k < (metal ? 6 : 5); ++k) {
        double m = k < 5 ? mass[u] + (k - 2) * MASS_H - z * ELECTRON
                         : mass[u] + (a == 4 ? 38.963158 : 22.989218);
        auto lo = std::lower_bound(sp.begin(), sp.end(), std::make_pair(float(m - tol), -1));
        for (auto p = lo; p != sp.end() && p->first <= m + tol; ++p) {
          match[s].push_back({uint16_t(p->second), uint16_t(u), uint8_t(k)});
          keep[u] = 1;
        }
      }
  }
  // keep ancestors of matched nodes: children have strictly fewer atoms
  std::vector<int> order(N);
  for (int i = 0; i < N; ++i) order[i] = i;
  // atom count (a strict subset has strictly fewer atoms) is a topological order
  std::vector<int> atomsN(N, 0);
  for (int u = 0; u < N; ++u)
    for (int w = 0; w < W; ++w) atomsN[u] += std::popcount(bits[size_t(u) * W + w]);
  std::stable_sort(order.begin(), order.end(), [&](int x, int y) { return atomsN[x] > atomsN[y]; });
  for (int i = N - 1; i >= 0; --i) {
    int u = order[i];
    if (keep[u]) continue;
    for (auto& e : tmpE[u]) if (keep[e.child]) { keep[u] = 1; break; }
  }
  // depth-cap nodes that are kept: count their events for K
  if (!O.leafObserve)
    for (int u = 0; u < N; ++u)
      if (keep[u] && depth[u] >= O.depth) {
        std::copy(bits.begin() + size_t(u) * W, bits.begin() + size_t(u + 1) * W, cur.begin());
        B.cuts(cur.data(), [&](int cls, const uint64_t* A, const uint64_t* Bb, int bnd) {
          ++cnt[u][cls];
          if (bnd < 0 || !QC.useDir) return;
          float qdq, qeA, qeB;
          qmEvent(bnd, cls, B.heavyCount(A), B.heavyCount(Bb), qdq, qeA, qeB);
          s1[u][cls] += qdq;
          s2[u][cls] += qdq * qdq;
        });
      }
  // compact, topological order
  std::vector<int> newId(N, -1);
  Dag D;
  int K = 0;
  for (int u : order) if (keep[u]) newId[u] = K++;
  if (K == 0 || newId[0] != 0) {
    // root always kept first (it is the largest); if nothing matched keep the root alone
    std::fill(newId.begin(), newId.end(), -1);
    K = 0;
    keep[0] = 1;
    for (int u : order) if (keep[u]) newId[u] = K++;
  }
  D.depth.resize(K);
  D.cnt.assign(size_t(K) * NCLSX, 0);
  if (QC.useDir) { D.s1.assign(size_t(K) * NCLSX, 0.f); D.s2.assign(size_t(K) * NCLSX, 0.f); }
  D.eStart.assign(K + 1, 0);
  for (int u : order) {
    int v = newId[u];
    if (v < 0) continue;
    D.depth[v] = depth[u];
    for (int c = 0; c < NCLSX; ++c) D.cnt[size_t(v) * NCLSX + c] = cnt[u][c];
    if (QC.useDir)
      for (int c = 0; c < NCLSX; ++c) { D.s1[size_t(v) * NCLSX + c] = s1[u][c]; D.s2[size_t(v) * NCLSX + c] = s2[u][c]; }
    for (auto e : tmpE[u])
      if (newId[e.child] >= 0) { e.child = uint16_t(newId[e.child]); D.edges.push_back(e); }
    D.eStart[v + 1] = uint32_t(D.edges.size());
  }
  D.match.resize(specs.size());
  long long nm = 0;
  for (size_t s = 0; s < specs.size(); ++s) {
    for (auto mt : match[s])
      if (newId[mt.node] >= 0) { mt.node = uint16_t(newId[mt.node]); D.match[s].push_back(mt); }
    nm += D.match[s].size();
  }
  st.nodes += N; st.kept += K; st.edges += D.edges.size(); st.matches += nm; st.mols += 1; st.capped += capped;
  return D;
}

// Observed fraction of every kept node for one charge mode (0 positive,
// 1 negative, 2 metal adduct): one topological pass of the absorbing chain.
void dagObserve(const Dag& D, const Params& P, int mode, double dof, std::vector<double>& flow,
                std::vector<double>& obs) {
  const size_t N = D.depth.size();
  flow.assign(N, 0.0);
  obs.assign(N, 0.0);
  if (!N) return;
  flow[0] = 1.0;
  const double s = std::exp(-P.E[INTACT]);
  double rate[8][NCLSX];
  for (int d = 0; d < 8; ++d)
    for (int c = 0; c < NCLSX; ++c) {
      double e = c < NCLS ? P.E[c]
                          : (c == RING1 || c == RING2 ? P.E[RING] : P.E[RINGAR]) +
                                P.dRing[(c == RING2 || c == RINGAR2) ? 1 : 0];
      rate[d][c] = c == INTACT ? 0.0 : std::exp(-e - dof - P.E2 * d);
    }
  const double beta = mode == 1 ? P.betaNeg : P.betaPos;
  // QM terms (kappa, gDir per 10 kcal); edge dq / ddir are stored in 0.1 kcal
  const bool useQ = !D.s1.empty();
  const double kq = 0.1 * P.kappa, gq = 0.01 * P.gDir;
  // voltage drop: energy each cleavage class spends, and U per node as the
  // flow-weighted mean over incoming paths (accumulated as flow * U)
  const bool vdrop = P.vLambda != 0.f;
  double spend[NCLSX];
  if (vdrop)
    for (int c = 0; c < NCLSX; ++c) {
      double e = c < NCLS ? P.E[c] : (c == RING1 || c == RING2 ? P.E[RING] : P.E[RINGAR]);
      spend[c] = std::log1p(std::exp(e + P.vC));  // softplus
    }
  static thread_local std::vector<double> flowU;
  if (vdrop) flowU.assign(N, 0.0);
  for (size_t n = 0; n < N; ++n) {
    if (flow[n] <= 0.0) continue;
    const int d = std::min<int>(D.depth[n], 7);
    const uint16_t* c = &D.cnt[n * NCLSX];
    const double U = vdrop ? flowU[n] / flow[n] : 0.0;
    const double slow = vdrop ? std::exp(-P.vLambda * U) : 1.0;
    double K = 0.0;
    if (useQ) {
      // sum_e exp(-kq dq_e) over the class events, to second order in kq dq
      const float* a1 = &D.s1[n * NCLSX];
      const float* a2 = &D.s2[n * NCLSX];
      for (int k = 1; k < NCLSX; ++k)
        if (c[k]) K += std::max(0.05 * c[k], double(c[k]) - kq * a1[k] + 0.5 * kq * kq * a2[k]) * rate[d][k];
    } else
      for (int k = 1; k < NCLSX; ++k) K += double(c[k]) * rate[d][k];
    K *= slow;
    const double f = flow[n] / (s + K);
    obs[n] = s * f;
    for (uint32_t e = D.eStart[n]; e < D.eStart[n + 1]; ++e) {
      const DagEdge& ed = D.edges[e];
      double share = mode == 2 ? 0.5
                               : 1.0 / (1.0 + std::exp(-beta * 0.01 * double(mode == 1 ? ed.dneg : ed.dpos) -
                                                       (mode == 0 ? gq * double(ed.ddir) : 0.0)));
      double add = f * rate[d][ed.cls] * slow * share * (useQ ? std::exp(-kq * 0.1 * double(ed.dq)) : 1.0);
      flow[ed.child] += add;
      if (vdrop) flowU[ed.child] += add * (U + spend[ed.cls]);
    }
  }
}

// Background-mixture log-likelihood (as phys_rank's loglik) from the
// precomputed peak <-> (node, H shift) matches; w = sqrt-intensity weights
// already normalized to sum 1.
float dagLL(const std::vector<Match>& mt, const std::vector<double>& obs, const double* wk, double tot,
            const float* w, int n, float tol, float eps, float range, std::vector<double>& m) {
  m.assign(size_t(n), 0.0);
  for (const Match& x : mt) m[x.peak] += obs[x.node] * wk[x.kind];
  const double bg = eps * 2.0 * tol / std::max(1.0f, range);
  double ll = 0.0;
  for (int i = 0; i < n; ++i)
    if (w[i] > 0.f) ll += double(w[i]) * std::log((1.0 - eps) * m[i] / tot + bg);
  return float(ll);
}

// ---- data ----------------------------------------------------------------

struct Query {
  std::string name;
  std::vector<int> spectra;
  std::vector<int> cand;     // pool indices
  std::vector<Dag> dags;     // pruned fragment DAG per candidate (matched against this query's spectra)
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
  if (argc < 7) {
    std::fprintf(stderr, "usage: phys_dag fit|eval <dir> <qry> <pool_stem> <ppm> <params> [out.csv]\n");
    return 1;
  }
  DagOpts dopt;
  auto envi = [](const char* k, int def) { const char* v = std::getenv(k); return v && *v ? std::atoi(v) : def; };
  dopt.depth = envi("DAG_DEPTH", 2);
  dopt.maxNodes = std::min(65000, envi("DAG_MAXNODES", 2000));
  dopt.leafObserve = envi("DAG_LEAF", 0) != 0;
  const unsigned nThreads = unsigned(envi("DAG_THREADS", int(std::max(1u, std::thread::hardware_concurrency()))));
  std::printf("DAG depth %d, max nodes %d, leaf %s, threads %u\n", dopt.depth, dopt.maxNodes,
              dopt.leafObserve ? "observe" : "lost", nThreads);
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
  // sqrt-intensity peak weights, normalized per spectrum (loglik's w)
  std::vector<float> wpk(it.size(), 0.f);
  for (const Head& h : head) {
    double ws = 0;
    for (int i = 0; i < h.n; ++i) ws += std::sqrt(std::max(0.0, double(it[h.off + i])));
    if (ws > 0)
      for (int i = 0; i < h.n; ++i) wpk[h.off + i] = float(std::sqrt(std::max(0.0, double(it[h.off + i]))) / ws);
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
  DagStats dstats;
  auto tEnum0 = std::chrono::steady_clock::now();
  for (unsigned t = 0; t < nThreads; ++t)
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
        std::vector<SpecView> sv(q.spectra.size());
        for (size_t i = 0; i < q.spectra.size(); ++i) {
          const Head& h = head[q.spectra[i]];
          sv[i].adduct = adductIdx(h.tag);
          for (int j = 0; j < h.n; ++j) sv[i].sorted.emplace_back(mz[h.off + j], j);
          std::sort(sv[i].sorted.begin(), sv[i].sorted.end());
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
          q.dags.push_back(buildDag(*m, sv, dopt, dstats, 0.01f));
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
  double tEnum = std::chrono::duration<double>(std::chrono::steady_clock::now() - tEnum0).count();
  {
    double nm = double(std::max<long long>(1, dstats.mols.load()));
    std::printf("DAG: %lld molecules, mean nodes %.1f, kept %.1f, edges %.1f, matches %.1f, node cap hit %lld\n",
                dstats.mols.load(), dstats.nodes / nm, dstats.kept / nm, dstats.edges / nm, dstats.matches / nm,
                dstats.capped.load());
    std::printf("enumeration wall %.1f s on %u threads = %.3f core-s per query\n", tEnum, nThreads,
                tEnum * nThreads / std::max<size_t>(1, Q.size()));
  }
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
    for (unsigned t = 0; t < nThreads; ++t)
      tt.emplace_back([&]() {
        std::vector<double> flow, obsM[3], mbuf;
        for (;;) {
          size_t qi = nx++;
          if (qi >= Q.size()) break;
          const Query& q = Q[qi];
          std::vector<float> sc(q.cand.size(), 0.f);
          float eps = std::exp(P.logEps);
          for (size_t c = 0; c < q.cand.size(); ++c) {
            const double dof = P.dofSlope * std::log(std::max(1, q.candN[c]) / 30.0);
            bool done[3] = {false, false, false};
            for (size_t si = 0; si < q.spectra.size(); ++si) {
              const Head& h = head[q.spectra[si]];
              int a = adductIdx(h.tag);
              bool neg = a >= 0 && a < 10 && ADDUCT_NEG[a];
              bool metal = a >= 0 && a < 10 && ADDUCT_METAL[a];
              int md = metal ? 2 : (neg ? 1 : 0);
              if (!done[md]) { dagObserve(q.dags[c], P, md, dof, flow, obsM[md]); done[md] = true; }
              const float* kw = neg ? P.kNeg : P.kPos;
              double wk[6], tot = 0;
              for (int k = 0; k < 5; ++k) { wk[k] = std::exp(kw[k]); tot += wk[k]; }
              wk[5] = 0.5;
              if (metal) tot += 0.5;
              bool empty = true;
              for (int i = 0; i < h.n; ++i) if (wpk[h.off + i] > 0.f) { empty = false; break; }
              sc[c] += empty ? -1e9f
                             : dagLL(q.dags[c].match[si], obsM[md], wk, tot, wpk.data() + h.off, h.n, 0.01f, eps,
                                     h.precursor, mbuf);
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
    // QM_FIT=i,j,...: coordinate search over these parameter indices only (kappa = N-2, gDir = N-1)
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
        float s = (i == NCLS - 1 || i == NCLS) ? step * 0.1f : step;  // betas (param 9, 10) live on a smaller scale
        // (phys_rank tests i == NCLS || NCLS + 1: betaPos gets the full step and kPos[-2] is never fitted)
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
    A = P; A.dRing[0] = A.dRing[1] = 0.f;
    std::printf("ablation: no ring-distance offsets %.4f\n", evaluate(A, nullptr));
    A = P; A.E2 = 0.f;
    std::printf("ablation: no depth penalty (E2 0) %.4f\n", evaluate(A, nullptr));
    std::printf("fitted:");
    const char* cn[NCLS] = {"intact", "acyl", "glycosidic", "C-het", "aryl-C", "C-C", "ring", "ring-ar", "other", "C-glycosidic"};
    for (int i = 0; i < NCLS; ++i) std::printf(" %s=%.3f", cn[i], P.E[i]);
    A = P; A.kappa = 0.f;
    std::printf("ablation: kappa 0 %.4f\n", evaluate(A, nullptr));
    A = P; A.gDir = 0.f;
    std::printf("ablation: gDir 0 %.4f\n", evaluate(A, nullptr));
    std::printf(" kappa=%.3f gDir=%.3f", P.kappa, P.gDir);
    std::printf(" E2=%.3f dofSlope=%.3f dRing1=%.3f dRing2=%.3f betaPos=%.3f betaNeg=%.3f\n", P.E2, P.dofSlope,
                P.dRing[0], P.dRing[1], P.betaPos, P.betaNeg);
    return 0;
  }

  if (!P.load(dir + "/" + paramPath)) std::printf("(params %s absent: defaults)\n", paramPath.c_str());
  std::vector<std::vector<float>> scores;
  auto tEv0 = std::chrono::steady_clock::now();
  double m = evaluate(P, &scores);
  std::printf("scoring wall %.2f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - tEv0).count());
  // [features.txt]: per-candidate features for fuse_fit (learned weights).
  // "Q <name> <truth index or -1> <n>" then n lines
  // "phys mlp rev libSim anaP inLib" (raw values; fuse_fit normalizes).
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
           << int(haveLib && libSpecOf.count(pname[q.cand[i]]) > 0) << "\n";
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
      std::vector<std::vector<float>> res(out ? Q.size() : 0);  // out may alias scores, read below
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
