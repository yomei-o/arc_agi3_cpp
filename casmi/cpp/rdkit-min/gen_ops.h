// gen_ops.h: structure generation from spectral analogs (C++ port of the
// Python prototype casmi/gen_recall.py; same transformation set and rules).
//
// A transformation is applied at every valid site of an (achiral) analog with
// RWMol edits, products are sanitized and keyed by non-isomeric canonical
// SMILES. One- and two-step chains whose total mass delta brings the analog
// into the query's neutral-mass window are applied (two-step chains in the
// sorted op order, as the prototype).
#pragma once
#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <GraphMol/GraphMol.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/PeriodicTable.h>
#include <GraphMol/RingInfo.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/SmilesParse/SmilesWrite.h>

namespace gen {

constexpr double HM = 1.00782503207;

inline double molMass(const RDKit::ROMol& m) {
  const RDKit::PeriodicTable* pt = RDKit::PeriodicTable::getTable();
  double s = 0;
  for (const auto a : m.atoms()) s += pt->getMostCommonIsotopeMass(a->getAtomicNum()) + a->getTotalNumHs(false) * HM;
  return s;
}

inline double atomMassH(const RDKit::Atom* a) {
  return RDKit::PeriodicTable::getTable()->getMostCommonIsotopeMass(a->getAtomicNum()) + a->getTotalNumHs(false) * HM;
}

inline std::string key(const RDKit::ROMol& m) { return RDKit::MolToSmiles(m, false); }

// site kinds
enum : unsigned { SO = 1, SN = 2, SS = 4, SCAR = 8, SC = 16 };
enum Kind { ATTACH, REMOVE, NOX, DENOX, HYD, DEHYD, HYDRATE, DEHYDRATE };

struct Op {
  std::string name;
  double delta;
  Kind kind;
  int group;  // ATTACH/REMOVE
};

struct Groups {
  std::vector<std::string> name;
  std::vector<std::unique_ptr<RDKit::ROMol>> mol;
  std::vector<unsigned> sites;
  std::vector<Op> ops;  // sorted by name (= the prototype's SINGLE order)
  std::unordered_map<std::string, int> opIdx;
  Groups() {
    struct G { const char* n; const char* s; unsigned k; };
    const G gs[] = {
        {"OH", "O", SC},
        {"Me", "C", SO | SN | SS | SCAR | SC},
        {"Ac", "C(C)=O", SO | SN},
        {"Hex", "C1OC(CO)C(O)C(O)C1O", SO | SCAR},
        {"dHex", "C1OC(C)C(O)C(O)C1O", SO},
        {"Pen", "C1OCC(O)C(O)C1O", SO},
        {"PenF", "C1OC(CO)C(O)C1O", SO},
        {"Api", "C1OCC(O)(CO)C1O", SO},
        {"GlcA", "C1OC(C(=O)O)C(O)C(O)C1O", SO},
        {"Mal", "C(=O)CC(=O)O", SO},
        {"SO3", "S(=O)(=O)O", SO},
        {"Prenyl", "CC=C(C)C", SCAR | SO},
    };
    for (const G& g : gs) {
      name.push_back(g.n);
      mol.emplace_back(RDKit::SmilesToMol(g.s));
      sites.push_back(g.k);
      int gi = int(name.size()) - 1;
      double d = molMass(*mol.back()) - 2 * HM;
      ops.push_back({std::string("+") + g.n, d, ATTACH, gi});
      ops.push_back({std::string("-") + g.n, -d, REMOVE, gi});
    }
    ops.push_back({"+NO", 15.9949146, NOX, -1});
    ops.push_back({"-NO", -15.9949146, DENOX, -1});
    ops.push_back({"+H2", 2 * HM, HYD, -1});
    ops.push_back({"-H2", -2 * HM, DEHYD, -1});
    ops.push_back({"+H2O", 18.0105647, HYDRATE, -1});
    ops.push_back({"-H2O", -18.0105647, DEHYDRATE, -1});
    std::sort(ops.begin(), ops.end(), [](const Op& a, const Op& b) { return a.name < b.name; });
    for (size_t i = 0; i < ops.size(); ++i) opIdx[ops[i].name] = int(i);
  }
};

inline const Groups& groups() {
  static Groups g;
  return g;
}

using MolP = std::unique_ptr<RDKit::RWMol>;

inline void addH(RDKit::Atom* a, int d) {
  if (a->getNoImplicit() || a->getNumExplicitHs() > 0)
    a->setNumExplicitHs(unsigned(std::max(0, int(a->getNumExplicitHs()) + d)));
}

inline MolP finish(RDKit::RWMol& rw) {
  try {
    RDKit::MolOps::sanitizeMol(rw);
  } catch (...) {
    return nullptr;
  }
  return MolP(new RDKit::RWMol(rw));
}

inline bool siteOK(const RDKit::Atom* a, unsigned k) {
  if (a->getTotalNumHs() == 0 || a->getFormalCharge() != 0) return false;
  int z = a->getAtomicNum();
  if (z == 6) {
    if (a->getIsAromatic()) return (k & SCAR) || ((k & SC) && !(k & SCAR));
    return (k & SC) != 0;
  }
  if (z == 8) return (k & SO) != 0;
  if (z == 7) return (k & SN) != 0;
  if (z == 16) return (k & SS) != 0;
  return false;
}

inline void attach(const RDKit::ROMol& m, int gi, std::vector<MolP>& out) {
  const Groups& G = groups();
  const RDKit::ROMol& g = *G.mol[gi];
  for (const auto a : m.atoms()) {
    if (!siteOK(a, G.sites[gi])) continue;
    RDKit::RWMol rw(m);
    unsigned base = rw.getNumAtoms();
    for (const auto ga : g.atoms()) rw.addAtom(new RDKit::Atom(*ga), true, true);
    for (const auto gb : g.bonds())
      rw.addBond(base + gb->getBeginAtomIdx(), base + gb->getEndAtomIdx(), gb->getBondType());
    addH(rw.getAtomWithIdx(a->getIdx()), -1);
    rw.addBond(a->getIdx(), base, RDKit::Bond::SINGLE);
    if (auto p = finish(rw)) out.push_back(std::move(p));
  }
}

inline void noxide(const RDKit::ROMol& m, std::vector<MolP>& out) {
  for (const auto a : m.atoms()) {
    if (a->getFormalCharge() != 0 || a->getTotalNumHs() != 0) continue;
    int z = a->getAtomicNum();
    bool allSingle = true;
    for (auto b : m.atomBonds(a)) if (b->getBondType() != RDKit::Bond::SINGLE) allSingle = false;
    if ((z == 7 && a->getDegree() == 3 && !a->getIsAromatic() && allSingle) ||
        (z == 7 && a->getIsAromatic() && a->getDegree() == 2)) {
      RDKit::RWMol rw(m);
      unsigned o = rw.addAtom(new RDKit::Atom(8), true, true);
      rw.getAtomWithIdx(a->getIdx())->setFormalCharge(1);
      rw.getAtomWithIdx(o)->setFormalCharge(-1);
      rw.addBond(a->getIdx(), o, RDKit::Bond::SINGLE);
      if (auto p = finish(rw)) out.push_back(std::move(p));
    } else if (z == 16 && a->getDegree() == 2 && !a->getIsAromatic()) {
      RDKit::RWMol rw(m);
      unsigned o = rw.addAtom(new RDKit::Atom(8), true, true);
      rw.addBond(a->getIdx(), o, RDKit::Bond::DOUBLE);
      if (auto p = finish(rw)) out.push_back(std::move(p));
    }
  }
}

inline void denoxide(const RDKit::ROMol& m, std::vector<MolP>& out) {
  for (const auto b : m.bonds()) {
    const RDKit::Atom* x = b->getBeginAtom();
    const RDKit::Atom* y = b->getEndAtom();
    for (int r = 0; r < 2; ++r) {
      const RDKit::Atom* n = r ? y : x;
      const RDKit::Atom* o = r ? x : y;
      if (n->getAtomicNum() == 7 && n->getFormalCharge() == 1 && o->getAtomicNum() == 8 &&
          o->getFormalCharge() == -1 && o->getDegree() == 1) {
        RDKit::RWMol rw(m);
        rw.getAtomWithIdx(n->getIdx())->setFormalCharge(0);
        rw.removeAtom(o->getIdx());
        if (auto p = finish(rw)) out.push_back(std::move(p));
      }
    }
  }
}

// atoms reachable from start without crossing bond `banned`
inline void sideAtoms(const RDKit::ROMol& m, int start, unsigned banned, std::vector<int>& side,
                      std::vector<char>& seen, size_t limit) {
  side.clear();
  seen.assign(m.getNumAtoms(), 0);
  seen[start] = 1;
  side.push_back(start);
  for (size_t k = 0; k < side.size(); ++k) {
    if (side.size() > limit) return;
    for (auto b : m.atomBonds(m.getAtomWithIdx(side[k]))) {
      if (b->getIdx() == banned) continue;
      int o = int(b->getOtherAtomIdx(side[k]));
      if (!seen[o]) { seen[o] = 1; side.push_back(o); }
    }
  }
}

// cut an acyclic single bond, drop one side (<= 16 atoms), cap with H; only
// products whose mass change is `delta` (+-0.002) are built
inline void removals(const RDKit::ROMol& m, double delta, std::vector<MolP>& out) {
  const RDKit::RingInfo* ri = m.getRingInfo();
  size_t n = m.getNumAtoms();
  std::vector<int> side;
  std::vector<char> seen;
  for (const auto b : m.bonds()) {
    if (ri->numBondRings(b->getIdx()) > 0 || b->getBondType() != RDKit::Bond::SINGLE) continue;
    int i = int(b->getBeginAtomIdx()), j = int(b->getEndAtomIdx());
    for (int r = 0; r < 2; ++r) {
      int keep = r ? j : i, drop = r ? i : j;
      sideAtoms(m, drop, b->getIdx(), side, seen, 16);
      if (side.size() > 16 || side.size() + 1 >= n) continue;
      bool charged = false;
      double sm = 0;
      for (int k : side) {
        const RDKit::Atom* a = m.getAtomWithIdx(k);
        if (a->getFormalCharge()) charged = true;
        sm += atomMassH(a);
      }
      if (charged) continue;
      if (std::fabs((-sm + HM) - delta) > 0.002) continue;
      RDKit::RWMol rw(m);
      addH(rw.getAtomWithIdx(keep), 1);
      rw.removeBond(i, j);
      std::sort(side.begin(), side.end(), std::greater<int>());
      for (int k : side) rw.removeAtom(unsigned(k));
      if (auto p = finish(rw)) out.push_back(std::move(p));
    }
  }
}

inline void hydro(const RDKit::ROMol& m, std::vector<MolP>& out) {
  for (const auto b : m.bonds()) {
    if (b->getBondType() != RDKit::Bond::DOUBLE || b->getIsAromatic()) continue;
    const RDKit::Atom* x = b->getBeginAtom();
    const RDKit::Atom* y = b->getEndAtom();
    if (x->getFormalCharge() || y->getFormalCharge()) continue;
    if (x->getAtomicNum() == 16 || y->getAtomicNum() == 16) continue;
    RDKit::RWMol rw(m);
    rw.getBondWithIdx(b->getIdx())->setBondType(RDKit::Bond::SINGLE);
    addH(rw.getAtomWithIdx(x->getIdx()), 1);
    addH(rw.getAtomWithIdx(y->getIdx()), 1);
    if (auto p = finish(rw)) out.push_back(std::move(p));
  }
}

inline bool isCNO(int z) { return z == 6 || z == 7 || z == 8; }

inline void dehydro(const RDKit::ROMol& m, std::vector<MolP>& out) {
  for (const auto b : m.bonds()) {
    if (b->getBondType() != RDKit::Bond::SINGLE || b->getIsAromatic()) continue;
    const RDKit::Atom* x = b->getBeginAtom();
    const RDKit::Atom* y = b->getEndAtom();
    if (x->getIsAromatic() || y->getIsAromatic() || x->getTotalNumHs() == 0 || y->getTotalNumHs() == 0) continue;
    if (x->getFormalCharge() || y->getFormalCharge()) continue;
    if (!isCNO(x->getAtomicNum()) || !isCNO(y->getAtomicNum())) continue;
    if (x->getAtomicNum() != 6 && y->getAtomicNum() != 6) continue;
    RDKit::RWMol rw(m);
    addH(rw.getAtomWithIdx(x->getIdx()), -1);
    addH(rw.getAtomWithIdx(y->getIdx()), -1);
    rw.getBondWithIdx(b->getIdx())->setBondType(RDKit::Bond::DOUBLE);
    if (auto p = finish(rw)) out.push_back(std::move(p));
  }
}

inline void hydrate(const RDKit::ROMol& m, std::vector<MolP>& out) {
  const RDKit::RingInfo* ri = m.getRingInfo();
  for (const auto b : m.bonds()) {
    const RDKit::Atom* x = b->getBeginAtom();
    const RDKit::Atom* y = b->getEndAtom();
    if (b->getBondType() == RDKit::Bond::DOUBLE && !b->getIsAromatic() && x->getAtomicNum() == 6 &&
        y->getAtomicNum() == 6) {
      for (int r = 0; r < 2; ++r) {
        const RDKit::Atom* c = r ? y : x;
        const RDKit::Atom* o2 = r ? x : y;
        RDKit::RWMol rw(m);
        rw.getBondWithIdx(b->getIdx())->setBondType(RDKit::Bond::SINGLE);
        unsigned o = rw.addAtom(new RDKit::Atom(8), true, true);
        rw.addBond(c->getIdx(), o, RDKit::Bond::SINGLE);
        addH(rw.getAtomWithIdx(o2->getIdx()), 1);
        if (auto p = finish(rw)) out.push_back(std::move(p));
      }
    } else if (ri->numBondRings(b->getIdx()) > 0 && b->getBondType() == RDKit::Bond::SINGLE && !b->getIsAromatic()) {
      for (int r = 0; r < 2; ++r) {
        const RDKit::Atom* c = r ? y : x;
        const RDKit::Atom* h = r ? x : y;
        int zh = h->getAtomicNum();
        if (c->getAtomicNum() == 6 && (zh == 8 || zh == 7) && !h->getIsAromatic()) {
          RDKit::RWMol rw(m);
          rw.removeBond(x->getIdx(), y->getIdx());
          unsigned o = rw.addAtom(new RDKit::Atom(8), true, true);
          rw.addBond(c->getIdx(), o, RDKit::Bond::SINGLE);
          addH(rw.getAtomWithIdx(h->getIdx()), 1);
          if (auto p = finish(rw)) out.push_back(std::move(p));
        }
      }
    }
  }
}

inline void dehydrate(const RDKit::ROMol& m, std::vector<MolP>& out) {
  std::vector<int> dist;
  for (const auto o : m.atoms()) {
    if (o->getAtomicNum() != 8 || o->getDegree() != 1 || o->getTotalNumHs() != 1) continue;
    const RDKit::Bond* cb = *m.atomBonds(o).begin();
    const RDKit::Atom* c = cb->getOtherAtom(o);
    if (c->getAtomicNum() != 6 || c->getIsAromatic() || cb->getBondType() != RDKit::Bond::SINGLE) continue;
    // elimination
    for (auto bb : m.atomBonds(c)) {
      const RDKit::Atom* nb = bb->getOtherAtom(c);
      if (nb == o || nb->getAtomicNum() != 6 || nb->getIsAromatic() || nb->getTotalNumHs() == 0) continue;
      if (bb->getBondType() != RDKit::Bond::SINGLE) continue;
      RDKit::RWMol rw(m);
      rw.getBondWithIdx(bb->getIdx())->setBondType(RDKit::Bond::DOUBLE);
      addH(rw.getAtomWithIdx(nb->getIdx()), -1);
      rw.removeAtom(o->getIdx());
      if (auto p = finish(rw)) out.push_back(std::move(p));
    }
    // cyclisation: OH + X-H (X = O/N, 4-6 bonds away) -> ring C-X
    size_t n = m.getNumAtoms();
    dist.assign(n, -1);
    std::vector<int> qq{int(c->getIdx())};
    dist[c->getIdx()] = 0;
    for (size_t k = 0; k < qq.size(); ++k) {
      if (dist[qq[k]] >= 6) continue;
      for (auto b : m.atomBonds(m.getAtomWithIdx(qq[k]))) {
        int t = int(b->getOtherAtomIdx(qq[k]));
        if (dist[t] < 0) { dist[t] = dist[qq[k]] + 1; qq.push_back(t); }
      }
    }
    for (const auto x : m.atoms()) {
      int zx = x->getAtomicNum();
      if ((zx != 8 && zx != 7) || x->getTotalNumHs() == 0 || x == o || x->getIsAromatic()) continue;
      int d = dist[x->getIdx()];
      if (d < 4 || d > 6) continue;
      RDKit::RWMol rw(m);
      addH(rw.getAtomWithIdx(x->getIdx()), -1);
      rw.addBond(c->getIdx(), x->getIdx(), RDKit::Bond::SINGLE);
      rw.removeAtom(o->getIdx());
      if (auto p = finish(rw)) out.push_back(std::move(p));
    }
  }
}

inline void apply(const RDKit::ROMol& m, const Op& op, std::vector<MolP>& out) {
  switch (op.kind) {
    case ATTACH: attach(m, op.group, out); break;
    case REMOVE: removals(m, op.delta, out); break;
    case NOX: noxide(m, out); break;
    case DENOX: denoxide(m, out); break;
    case HYD: hydro(m, out); break;
    case DEHYD: dehydro(m, out); break;
    case HYDRATE: hydrate(m, out); break;
    case DEHYDRATE: dehydrate(m, out); break;
  }
}

// One generated structure, aggregated over all analogs / chains producing it.
struct Prod {
  MolP mol;
  float sim = 0.f;       // best source-analog similarity
  int rank = 1 << 30;    // best source-analog rank
  int steps = 9;         // fewest steps
  int nSib = 0;          // products of the best (analog, chain): site ambiguity
  std::vector<int> sup;  // distinct source-analog ranks
  std::vector<std::string> chains;
};

// Chains (op index lists) whose total delta lands within any [lo, hi] window
// (windows on the neutral mass) from analog mass am.
inline void chainsFor(double am, const std::vector<std::pair<double, double>>& win,
                      std::vector<std::vector<int>>& out, bool twoStep) {
  const auto& ops = groups().ops;
  auto ok = [&](double m) {
    for (auto& w : win) if (m >= w.first && m <= w.second) return true;
    return false;
  };
  out.clear();
  for (size_t i = 0; i < ops.size(); ++i) if (ok(am + ops[i].delta)) out.push_back({int(i)});
  if (!twoStep) return;
  for (size_t i = 0; i < ops.size(); ++i)
    for (size_t j = i; j < ops.size(); ++j) {
      if (ops[i].name.substr(1) == ops[j].name.substr(1) && ops[i].name[0] != ops[j].name[0]) continue;
      if (ok(am + ops[i].delta + ops[j].delta)) out.push_back({int(i), int(j)});
    }
}

// Generate from one analog (stereo-free, sanitized). Products land in `prod`
// keyed by canonical SMILES. stepCap bounds each step's distinct products.
inline void generateFrom(const RDKit::ROMol& am, int rank, float sim, const std::vector<std::vector<int>>& chains,
                         std::unordered_map<std::string, Prod>& prod, size_t stepCap) {
  const auto& ops = groups().ops;
  std::unordered_map<int, std::vector<std::pair<std::string, std::shared_ptr<RDKit::RWMol>>>> step1;
  auto run1 = [&](const RDKit::ROMol& m, const Op& op,
                  std::vector<std::pair<std::string, std::shared_ptr<RDKit::RWMol>>>& res,
                  std::unordered_map<std::string, int>& seen) {
    std::vector<MolP> raw;
    apply(m, op, raw);
    for (auto& p : raw) {
      std::string k = key(*p);
      if (seen.emplace(k, 1).second) res.emplace_back(std::move(k), std::shared_ptr<RDKit::RWMol>(p.release()));
    }
  };
  for (const auto& ch : chains) {
    auto it = step1.find(ch[0]);
    if (it == step1.end()) {
      std::vector<std::pair<std::string, std::shared_ptr<RDKit::RWMol>>> res;
      std::unordered_map<std::string, int> seen;
      run1(am, ops[ch[0]], res, seen);
      it = step1.emplace(ch[0], std::move(res)).first;
    }
    std::vector<std::pair<std::string, std::shared_ptr<RDKit::RWMol>>> cur = it->second;
    for (size_t s = 1; s < ch.size(); ++s) {
      std::vector<std::pair<std::string, std::shared_ptr<RDKit::RWMol>>> nxt;
      std::unordered_map<std::string, int> seen;
      for (auto& c : cur) {
        run1(*c.second, ops[ch[s]], nxt, seen);
        if (nxt.size() > stepCap) break;
      }
      cur.swap(nxt);
    }
    std::string chName;
    for (size_t s = 0; s < ch.size(); ++s) chName += (s ? "/" : "") + ops[ch[s]].name;
    int nSib = int(cur.size());
    for (auto& c : cur) {
      Prod& e = prod[c.first];
      if (!e.mol) e.mol.reset(new RDKit::RWMol(*c.second));
      if (sim > e.sim || (sim == e.sim && nSib < e.nSib)) { e.sim = sim; e.nSib = nSib; }
      e.rank = std::min(e.rank, rank);
      e.steps = std::min(e.steps, int(ch.size()));
      if (std::find(e.sup.begin(), e.sup.end(), rank) == e.sup.end()) e.sup.push_back(rank);
      if (std::find(e.chains.begin(), e.chains.end(), chName) == e.chains.end()) e.chains.push_back(chName);
    }
  }
}

}  // namespace gen
