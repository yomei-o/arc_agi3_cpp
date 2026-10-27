// gl_feat.hpp: GLACIER input features from a SMILES with RDKit C++ (rdkit-min + GraphMol/PartialCharges) =
// ms_pred glacier TreeProcessor.featurize_tree + create_graphormer_input (pe_embed_k 14, embed_elem_group,
// multi_hop_max_dist 10) and gl_runner.canon (valid atoms only, RemoveHs, <= 100 heavy atoms, MolToSmiles).
#pragma once
#include <GraphMol/GraphMol.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/PartialCharges/GasteigerCharges.h>
#include <GraphMol/PeriodicTable.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/SmilesParse/SmilesWrite.h>

#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "gl_model.hpp"

namespace gl {

// ms_pred.common tables (glacier_hp.json)
static const int VALID_ATOM_NUM[18] = {6, 7, 15, 8, 16, 14, 53, 1, 17, 9, 35, 5, 34, 26, 27, 33, 11, 19};
static const char* VALID_ELEMENTS[18] = {"C", "N", "P", "O", "S", "Si", "I", "H", "Cl", "F",
                                         "Br", "B", "Se", "Fe", "Co", "As", "Na", "K"};
static const double ELEMENT_MASS[18] = {12.0,        14.003074,   30.97376163, 15.99491462, 31.972071,  27.97692653,
                                        126.904473,  1.007825032, 34.96885268, 18.99840322, 78.9183371, 11.0093054,
                                        79.9165213,  55.9349375,  58.933195,   74.9215965,  22.98976928, 38.96370668};
static const int ELEMENT_GROUP[18] = {4, 3, 3, 5, 5, 4, 6, 0, 6, 6, 6, 2, 5, 7, 7, 3, 1, 1};
static const int HYB[5] = {2, 3, 4, 6, 7};    // SP, SP2, SP3, SP3D, SP3D2
static const int RING_SIZES[6] = {3, 4, 5, 6, 7, 8};
static const int BOND_TYPES[4] = {1, 2, 3, 12};  // SINGLE DOUBLE TRIPLE AROMATIC
constexpr int EDGE_F = 28, PE_K = 14;

inline int elem_index(int z) {
  for (int e = 0; e < 18; e++)
    if (VALID_ATOM_NUM[e] == z) return e;
  return -1;
}

// gl_runner.canon -> canonical SMILES or "" (rejected)
inline std::string canon(const std::string& smi) {
  std::unique_ptr<RDKit::RWMol> m;
  try {
    m.reset(RDKit::SmilesToMol(smi));
  } catch (...) {
    return "";
  }
  if (!m) return "";
  for (const RDKit::Atom* a : m->atoms())
    if (elem_index(a->getAtomicNum()) < 0) return "";
  try {
    RDKit::MolOps::removeHs(*m);
  } catch (...) {
    return "";
  }
  if (m->getNumAtoms() > 100) return "";
  return RDKit::MolToSmiles(*m);
}

// Featurise a canonical SMILES (the runner featurises its canonical form). Returns false on failure.
inline bool featurize(const std::string& csmi, MolFeat& F) {
  std::unique_ptr<RDKit::RWMol> mol;
  try {
    mol.reset(RDKit::SmilesToMol(csmi));
    if (!mol) return false;
    RDKit::MolOps::removeHs(*mol);
  } catch (...) {
    return false;
  }
  const int n = (int)mol->getNumAtoms();
  if (n == 0) return false;
  F = MolFeat();
  F.n = n;
  try {
    RDKit::computeGasteigerCharges(*mol, 12, false);
  } catch (...) {
  }
  std::vector<std::vector<int>> sssr;
  RDKit::MolOps::symmetrizeSSSR(*mol, sssr);
  std::vector<int> atom_rc(n, 0);
  std::map<std::pair<int, int>, int> bond_rc;
  for (const RDKit::Bond* b : mol->bonds()) {
    int i = (int)b->getBeginAtomIdx(), j = (int)b->getEndAtomIdx();
    bond_rc[{std::min(i, j), std::max(i, j)}] = 0;
  }
  for (const auto& ring : sssr) {
    int L = (int)ring.size();
    for (int a : ring) atom_rc[a]++;
    for (int k = 0; k < L; k++) {
      int a1 = ring[k], a2 = ring[(k + 1) % L];
      auto it = bond_rc.find({std::min(a1, a2), std::max(a1, a2)});
      if (it != bond_rc.end()) it->second++;
    }
  }
  const RDKit::RingInfo* ri = mol->getRingInfo();
  // ---- node features (80 + PE 14)
  const int NF0 = 74;
  F.nfeat = NF0 + PE_K;
  F.x.assign((size_t)n * F.nfeat, 0.f);
  F.degree.resize(n);
  for (int i = 0; i < n; i++) {
    const RDKit::Atom* a = mol->getAtomWithIdx(i);
    float* r = &F.x[(size_t)i * F.nfeat];
    int c = 0;
    int z = a->getAtomicNum(), e = elem_index(z);
    for (int t = 0; t < 18; t++) r[c++] = (float)(z == VALID_ATOM_NUM[t]);
    r[c++] = (float)(e < 0);
    int nH = (int)a->getTotalNumHs();
    int deg = (int)a->getTotalDegree() - nH;
    for (int t = 0; t <= 5; t++) r[c++] = (float)(deg == t);
    r[c++] = (float)(deg > 5);
    int fc = a->getFormalCharge();
    for (int t = -2; t <= 2; t++) r[c++] = (float)(fc == t);
    r[c++] = (float)(std::abs(fc) > 2);
    int hy = (int)a->getHybridization();
    bool hin = false;
    for (int t = 0; t < 5; t++) {
      r[c++] = (float)(hy == HYB[t]);
      hin |= hy == HYB[t];
    }
    r[c++] = (float)!hin;
    r[c++] = (float)a->getIsAromatic();
    for (int t = 0; t <= 4; t++) r[c++] = (float)(nH == t);
    r[c++] = (float)(nH > 4);
    bool inr = ri->numAtomRings(i) != 0;
    r[c++] = (float)!inr;
    r[c++] = (float)inr;
    for (int rs : RING_SIZES) {
      bool b = ri->isAtomInRingOfSize(i, rs);
      r[c++] = (float)b;
      r[c++] = (float)!b;
    }
    bool fused = atom_rc[i] >= 2;
    r[c++] = (float)!fused;
    r[c++] = (float)fused;
    int chi = (int)a->getChiralTag();
    for (int t = 0; t < 4; t++) r[c++] = (float)(chi == t);
    if (e >= 0) r[c + ELEMENT_GROUP[e]] = 1.f;
    c += 8;
    double q = 0.0;
    if (a->hasProp("_GasteigerCharge")) {
      try {
        q = a->getProp<double>("_GasteigerCharge");
      } catch (...) {
        q = 0.0;
      }
    }
    if (std::isnan(q) || std::isinf(q)) q = 0.0;
    r[c++] = (float)q;
    F.degree[i] = deg;
  }
  // ---- adjacency (bond orders) and random-walk PE: diag(P^k), P = row-normalised bond-order adjacency
  F.adj_bo.assign((size_t)n * n, 0.f);
  for (const RDKit::Bond* b : mol->bonds()) {
    int i = (int)b->getBeginAtomIdx(), j = (int)b->getEndAtomIdx();
    float v = (float)b->getBondTypeAsDouble();
    F.adj_bo[(size_t)i * n + j] = v;
    F.adj_bo[(size_t)j * n + i] = v;
  }
  {
    std::vector<float> P((size_t)n * n, 0.f), O, T((size_t)n * n);
    for (int i = 0; i < n; i++) {
      float s = 0.f;
      for (int j = 0; j < n; j++) s += F.adj_bo[(size_t)i * n + j];
      s += 1e-30f;
      for (int j = 0; j < n; j++) P[(size_t)i * n + j] = F.adj_bo[(size_t)i * n + j] / s;
    }
    O = P;
    for (int k = 0; k < PE_K; k++) {
      if (k > 0) {
        for (int i = 0; i < n; i++)
          for (int j = 0; j < n; j++) {
            float s = 0.f;
            for (int t = 0; t < n; t++) s += O[(size_t)i * n + t] * P[(size_t)t * n + j];
            T[(size_t)i * n + j] = s;
          }
        O.swap(T);
      }
      for (int i = 0; i < n; i++) F.x[(size_t)i * F.nfeat + NF0 + k] = O[(size_t)i * n + i];
    }
  }
  // ---- shortest paths (BFS; parent = lowest-index frontier neighbour) and multi-hop edge features
  std::vector<std::vector<uint8_t>> A(n, std::vector<uint8_t>(n, 0));
  for (const RDKit::Bond* b : mol->bonds()) {
    int i = (int)b->getBeginAtomIdx(), j = (int)b->getEndAtomIdx();
    A[i][j] = A[j][i] = 1;
  }
  F.spatial.assign((size_t)n * n, 1023);
  std::vector<int> parent((size_t)n * n, -1);
  for (int src = 0; src < n; src++) {
    std::vector<uint8_t> vis(n, 0), fr(n, 0);
    vis[src] = fr[src] = 1;
    F.spatial[(size_t)src * n + src] = 0;
    parent[(size_t)src * n + src] = src;
    int depth = 0;
    while (true) {
      std::vector<uint8_t> nx(n, 0);
      bool any = false;
      for (int j = 0; j < n; j++) {
        if (vis[j]) continue;
        for (int i = 0; i < n; i++)
          if (fr[i] && A[i][j]) {
            nx[j] = 1;
            any = true;
            break;
          }
      }
      if (!any) break;
      depth++;
      for (int j = 0; j < n; j++)
        if (nx[j]) {
          F.spatial[(size_t)src * n + j] = depth;
          for (int i = 0; i < n; i++)
            if (fr[i] && A[i][j]) {
              parent[(size_t)src * n + j] = i;
              break;
            }
          vis[j] = 1;
        }
      fr = nx;
    }
  }
  std::vector<float> bf((size_t)n * n * EDGE_F, 0.f);
  for (const RDKit::Bond* b : mol->bonds()) {
    int i = (int)b->getBeginAtomIdx(), j = (int)b->getEndAtomIdx(), bi = (int)b->getIdx();
    float r[EDGE_F] = {0};
    int c = 0, bt = (int)b->getBondType();
    bool tin = false;
    for (int t = 0; t < 4; t++) {
      r[c++] = (float)(bt == BOND_TYPES[t]);
      tin |= bt == BOND_TYPES[t];
    }
    r[c++] = (float)!tin;
    bool conj = b->getIsConjugated();
    r[c++] = (float)!conj;
    r[c++] = (float)conj;
    bool inr = ri->numBondRings(bi) != 0;
    r[c++] = (float)!inr;
    r[c++] = (float)inr;
    for (int rs : RING_SIZES) {
      bool v = ri->isBondInRingOfSize(bi, rs);
      r[c++] = (float)v;
      r[c++] = (float)!v;
    }
    bool fused = bond_rc[{std::min(i, j), std::max(i, j)}] >= 2;
    r[c++] = (float)!fused;
    r[c++] = (float)fused;
    int st = (int)b->getStereo();
    for (int t = 0; t < 4; t++) r[c++] = (float)(st == t);
    r[c++] = (float)(st > 3);
    std::memcpy(&bf[((size_t)i * n + j) * EDGE_F], r, sizeof r);
    std::memcpy(&bf[((size_t)j * n + i) * EDGE_F], r, sizeof r);
  }
  F.edge_input.assign((size_t)n * n * MAXDIST * EDGE_F, 0.f);
  for (int src = 0; src < n; src++)
    for (int tgt = 0; tgt < n; tgt++) {
      if (tgt == src || parent[(size_t)src * n + tgt] < 0) continue;
      std::vector<int> path{tgt};
      int cur = tgt;
      while (cur != src) {
        cur = parent[(size_t)src * n + cur];
        path.push_back(cur);
      }
      std::reverse(path.begin(), path.end());
      int hops = std::min((int)path.size() - 1, MAXDIST);
      for (int h = 0; h < hops; h++)
        std::memcpy(&F.edge_input[(((size_t)src * n + tgt) * MAXDIST + h) * EDGE_F],
                    &bf[((size_t)path[h] * n + path[h + 1]) * EDGE_F], EDGE_F * sizeof(float));
    }
  // ---- masses / hydrogens / formulas (FragmentEngine on the kekulised molecule; same atom order)
  F.masses.resize(n);
  F.atom_hs.resize(n);
  F.atom_form.assign((size_t)n * NFORM, 0.f);
  F.root_form.assign(NFORM, 0.f);
  F.total_hs = 0;
  const RDKit::PeriodicTable* pt = RDKit::PeriodicTable::getTable();
  for (int i = 0; i < n; i++) {
    const RDKit::Atom* a = mol->getAtomWithIdx(i);
    int e = elem_index(a->getAtomicNum());
    int h = (int)a->getTotalNumHs();
    double w = a->getIsotope() == 0 ? (e >= 0 ? ELEMENT_MASS[e] : pt->getMostCommonIsotopeMass(a->getAtomicNum()))
                                    : pt->getMassForIsotope(a->getAtomicNum(), a->getIsotope());
    F.masses[i] = (float)(h * ELEMENT_MASS[7] + w);
    F.atom_hs[i] = h;
    F.total_hs += h;
    if (e >= 0) {
      F.atom_form[(size_t)i * NFORM + e] += 1.f;
      F.root_form[e] += 1.f;
    }
    F.atom_form[(size_t)i * NFORM + 7] += (float)h;
    F.root_form[7] += (float)h;
  }
  return true;
}

// adduct tables (ms_pred.common.ion2onehot_pos / ion2mass / is_positive_adduct) for the adducts the runner covers
inline bool cond_for(const std::string& adduct, const std::string& instr, float ce, Cond& C) {
  static const std::map<std::string, std::pair<int, double>> AD = {
      {"[M+H]+", {0, 1.007276452}},      {"[M+Na]+", {1, 22.9892207}},    {"[M+K]+", {2, 38.9631581}},
      {"[M-H2O+H]+", {3, -17.003288232}}, {"[M+H-H2O]+", {3, -17.003288232}}, {"[M+NH4]+", {4, 18.033825548}},
      {"[M]+", {5, -0.00054858}},         {"[M-H]-", {10, -1.007276452}},  {"[M+Cl]-", {11, 34.96940126}},
      {"[M+CHO2]-", {9, 44.998202852}}};
  auto it = AD.find(adduct);
  if (it == AD.end()) return false;
  C.adduct = it->second.first;
  C.adduct_mass = (float)it->second.second;
  bool pos = adduct.back() == '+';
  C.electron = pos ? -0.00054858f : 0.00054858f;
  C.instrument = instr == "Orbitrap" ? 0 : instr == "QTOF" ? 1 : instr == "IT-FT" ? 2 : 3;
  C.ce = ce;
  return true;
}

}  // namespace gl
