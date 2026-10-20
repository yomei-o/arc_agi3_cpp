// fp3: extended collision-free "vocabulary" features for the reverse model v3.
// Every feature is a 64-bit hash tagged by its kind; a vocabulary (same file
// layout as fp2: magic 0x32565046, rE, rF, n, hashes) picks the bits. rE = 300
// marks an fp3 vocabulary (rF = the fp3 feature version, FP3_VERSION).
//
// Kinds (all own definitions; aimed at separating ISOMERS inside a mass window):
//   0/1  fp2 ECFP-like / FCFP-like environments r0..3 (fp2.h, unchanged hashes)
//   2    atom pairs: (typed atom i, typed atom j, topological distance 1..12)
//   3    ring substitution patterns: per ring, the cyclic sequence of (ring-atom
//        element, exocyclic substituent class) canonical over rotation and
//        reflection, plus a coarse variant (size, aromatic, positions occupied)
//   4    ring systems: fused-ring block = sorted ring sizes + hetero counts +
//        aromatic ring count; and the number of ring systems
//   5    counts: ECFP r0 / r1 environment multiplicities >= 2,3,4,6 and element
//        counts / H-donor / acceptor counts >= thresholds
//   6    linear paths of 2..6 bonds (element+aromatic, bond orders), direction-canonical
#pragma once

#include <algorithm>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include <GraphMol/GraphMol.h>
#include <GraphMol/RingInfo.h>

#include "fp2.h"

namespace fp3 {

constexpr int FP3_VERSION = 1;
using fp2::mix;

inline uint64_t tagk(uint64_t h, int kind) { return mix(mix(h, 0xF3F3ull + uint64_t(kind) * 0x1001ull), 77); }

// coarse atom type for pairs / paths
inline uint64_t atype(const RDKit::ROMol& mol, const RDKit::Atom* a, const RDKit::RingInfo* ri) {
  uint64_t h = mix(0x51ull, uint64_t(a->getAtomicNum()));
  h = mix(h, uint64_t(a->getDegree()));
  h = mix(h, uint64_t(std::min(3u, a->getTotalNumHs())));
  h = mix(h, uint64_t(a->getIsAromatic()));
  h = mix(h, uint64_t(ri->numAtomRings(a->getIdx()) > 0));
  (void)mol;
  return h;
}

// exocyclic substituent class of ring atom i relative to ring set `inRing`
inline int subClass(const RDKit::ROMol& mol, int i, const std::vector<char>& thisRing, const RDKit::RingInfo* ri) {
  int cls = 0;
  for (const auto& bnd : mol.atomBonds(mol.getAtomWithIdx(i))) {
    int j = int(bnd->getOtherAtomIdx(i));
    if (thisRing[j]) continue;
    const RDKit::Atom* o = mol.getAtomWithIdx(j);
    int c;
    if (ri->numAtomRings(j) > 0 && ri->numAtomRings(i) > 1 && bnd->getIsAromatic()) c = 1;  // fused
    else if (ri->numAtomRings(j) > 0) c = 2;                                                 // ring substituent
    else {
      int z = o->getAtomicNum();
      bool dbl = bnd->getBondType() == RDKit::Bond::DOUBLE;
      if (z == 8) c = dbl ? 3 : (o->getTotalNumHs() > 0 ? 4 : 5);   // =O, OH, OR
      else if (z == 7) c = 6;
      else if (z == 6) {
        bool carbonyl = false;
        for (const auto& b2 : mol.atomBonds(o))
          if (b2->getBondType() == RDKit::Bond::DOUBLE && b2->getOtherAtom(o)->getAtomicNum() == 8) carbonyl = true;
        c = carbonyl ? 7 : (o->getDegree() == 1 ? 8 : 9);          // C=O, CH3, other C
      } else if (z == 9 || z == 17 || z == 35 || z == 53) c = 10;
      else if (z == 16) c = 11;
      else c = 12;
    }
    cls = std::max(cls, c);  // deterministic summary if several
    if (c == 1) cls = 1;
  }
  return cls;
}

inline uint64_t seqHash(const std::vector<uint64_t>& s) {
  uint64_t h = 0x9911ull;
  for (uint64_t x : s) h = mix(h, x);
  return h;
}
// canonical over rotation/reflection: min hash of the 2n sequences
inline uint64_t cycCanon(const std::vector<uint64_t>& s) {
  const size_t n = s.size();
  uint64_t best = ~0ull;
  std::vector<uint64_t> t(n);
  for (int dir = 0; dir < 2; ++dir)
    for (size_t r = 0; r < n; ++r) {
      for (size_t k = 0; k < n; ++k) t[k] = dir == 0 ? s[(r + k) % n] : s[(r + n - k) % n];
      best = std::min(best, seqHash(t));
    }
  return best;
}

// All feature hashes of the molecule (sorted, unique).
inline std::vector<uint64_t> features(const RDKit::ROMol& mol, int kindsMask = 0x7F) {
  std::vector<uint64_t> out;
  const int n = int(mol.getNumAtoms());
  const RDKit::RingInfo* ri = mol.getRingInfo();
  if (kindsMask & 3) {
    auto e = fp2::envs(mol, (kindsMask & 1) ? 3 : -1, (kindsMask & 2) ? 3 : -1);
    out.insert(out.end(), e.begin(), e.end());
  }
  std::vector<uint64_t> at(n);
  for (int i = 0; i < n; ++i) at[i] = atype(mol, mol.getAtomWithIdx(i), ri);
  if (kindsMask & 4) {  // atom pairs, BFS distances up to 12
    std::vector<int> dist(n);
    std::vector<int> qu(n);
    for (int s = 0; s < n; ++s) {
      std::fill(dist.begin(), dist.end(), -1);
      dist[s] = 0;
      int qh = 0, qt = 0;
      qu[qt++] = s;
      while (qh < qt) {
        int u = qu[qh++];
        if (dist[u] >= 12) continue;
        for (const auto& bnd : mol.atomBonds(mol.getAtomWithIdx(u))) {
          int v = int(bnd->getOtherAtomIdx(u));
          if (dist[v] < 0) { dist[v] = dist[u] + 1; qu[qt++] = v; }
        }
      }
      for (int t = s + 1; t < n; ++t)
        if (dist[t] > 0) {
          uint64_t a = at[s], b = at[t];
          if (a > b) std::swap(a, b);
          out.push_back(tagk(mix(mix(a, b), uint64_t(dist[t])), 2));
        }
    }
  }
  const auto& rings = ri->atomRings();
  if (kindsMask & 8) {  // ring substitution patterns
    std::vector<char> inR(n, 0);
    for (const auto& r : rings) {
      for (int a : r) inR[a] = 1;
      bool arom = true;
      std::vector<uint64_t> fine, coarse;
      for (int a : r) {
        const RDKit::Atom* A = mol.getAtomWithIdx(a);
        arom = arom && A->getIsAromatic();
        int sc = subClass(mol, a, inR, ri);
        fine.push_back(mix(uint64_t(A->getAtomicNum()), uint64_t(sc)));
        coarse.push_back(uint64_t(sc == 0 ? 0 : (sc == 1 ? 1 : 2)));
      }
      uint64_t base = mix(uint64_t(r.size()), uint64_t(arom));
      out.push_back(tagk(mix(mix(base, 1), cycCanon(fine)), 3));
      out.push_back(tagk(mix(mix(base, 2), cycCanon(coarse)), 3));
      for (int a : r) inR[a] = 0;
    }
  }
  if (kindsMask & 16) {  // ring systems (fused blocks: rings sharing a bond)
    const auto& brings = ri->bondRings();
    int R = int(rings.size());
    std::vector<int> comp(R, -1);
    int nc = 0;
    for (int i = 0; i < R; ++i) {
      if (comp[i] >= 0) continue;
      std::vector<int> st{i};
      comp[i] = nc;
      while (!st.empty()) {
        int u = st.back(); st.pop_back();
        for (int v = 0; v < R; ++v) {
          if (comp[v] >= 0) continue;
          bool share = false;
          for (int b1 : brings[u]) { for (int b2 : brings[v]) if (b1 == b2) { share = true; break; } if (share) break; }
          if (share) { comp[v] = nc; st.push_back(v); }
        }
      }
      ++nc;
    }
    for (int c = 0; c < nc; ++c) {
      std::vector<int> sizes;
      std::map<int, int> het;
      std::vector<char> seen(n, 0);
      int nArom = 0;
      for (int i = 0; i < R; ++i) {
        if (comp[i] != c) continue;
        sizes.push_back(int(rings[i].size()));
        bool ar = true;
        for (int a : rings[i]) {
          ar = ar && mol.getAtomWithIdx(a)->getIsAromatic();
          if (!seen[a]) { seen[a] = 1; int z = mol.getAtomWithIdx(a)->getAtomicNum(); if (z != 6) ++het[z]; }
        }
        nArom += ar;
      }
      std::sort(sizes.begin(), sizes.end());
      uint64_t h = 0x77ull;
      for (int s : sizes) h = mix(h, uint64_t(s));
      out.push_back(tagk(mix(h, 1), 4));
      h = mix(h, uint64_t(nArom));
      for (auto& kv : het) { h = mix(h, uint64_t(kv.first)); h = mix(h, uint64_t(kv.second)); }
      out.push_back(tagk(mix(h, 2), 4));
    }
    for (int k = 1; k <= std::min(nc, 6); ++k) out.push_back(tagk(mix(0x5151ull, uint64_t(k)), 4));
  }
  if (kindsMask & 32) {  // counts
    std::unordered_map<uint64_t, int> c0, c1;
    std::vector<uint64_t> h0(n);
    for (int i = 0; i < n; ++i) {
      const RDKit::Atom* a = mol.getAtomWithIdx(i);
      uint64_t h = mix(0x1234567ull, uint64_t(a->getAtomicNum()));
      h = mix(h, uint64_t(a->getDegree()));
      h = mix(h, uint64_t(a->getTotalNumHs()));
      h = mix(h, uint64_t(int64_t(a->getFormalCharge()) + 8));
      h = mix(h, uint64_t(ri->numAtomRings(i) > 0));
      h = mix(h, uint64_t(a->getIsAromatic()));
      h0[i] = h;
      ++c0[h];
    }
    for (int i = 0; i < n; ++i) {
      std::vector<std::pair<uint64_t, uint64_t>> nb;
      for (const auto& bnd : mol.atomBonds(mol.getAtomWithIdx(i)))
        nb.emplace_back(uint64_t(bnd->getBondType()), h0[bnd->getOtherAtomIdx(i)]);
      std::sort(nb.begin(), nb.end());
      uint64_t h = mix(1, h0[i]);
      for (auto& p : nb) { h = mix(h, p.first); h = mix(h, p.second); }
      ++c1[h];
    }
    const int TH[] = {2, 3, 4, 6};
    for (auto& kv : c0) for (int t : TH) if (kv.second >= t) out.push_back(tagk(mix(mix(kv.first, 0), uint64_t(t)), 5));
    for (auto& kv : c1) for (int t : TH) if (kv.second >= t) out.push_back(tagk(mix(mix(kv.first, 1), uint64_t(t)), 5));
    std::map<int, int> el;
    int don = 0, acc = 0, nArRing = 0;
    for (int i = 0; i < n; ++i) {
      const RDKit::Atom* a = mol.getAtomWithIdx(i);
      ++el[a->getAtomicNum()];
      uint64_t pc = fp2::pharmClass(mol, a);
      don += (pc & 1) != 0; acc += (pc & 2) != 0;
    }
    for (const auto& r : rings) {
      bool ar = true;
      for (int a : r) ar = ar && mol.getAtomWithIdx(a)->getIsAromatic();
      nArRing += ar;
    }
    const int TE[] = {1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40};
    for (auto& kv : el) for (int t : TE) if (kv.second >= t) out.push_back(tagk(mix(mix(0xE1ull, uint64_t(kv.first)), uint64_t(t)), 5));
    for (int t : TE) {
      if (don >= t) out.push_back(tagk(mix(0xD0ull, uint64_t(t)), 5));
      if (acc >= t) out.push_back(tagk(mix(0xACull, uint64_t(t)), 5));
      if (int(rings.size()) >= t) out.push_back(tagk(mix(0x81ull, uint64_t(t)), 5));
      if (nArRing >= t) out.push_back(tagk(mix(0x82ull, uint64_t(t)), 5));
    }
  }
  if (kindsMask & 64) {  // linear paths of 2..6 bonds
    std::vector<uint64_t> el(n);
    for (int i = 0; i < n; ++i) {
      const RDKit::Atom* a = mol.getAtomWithIdx(i);
      el[i] = uint64_t(a->getAtomicNum()) * 2 + uint64_t(a->getIsAromatic());
    }
    std::vector<int> path;
    std::vector<uint64_t> bt;
    std::vector<char> on(n, 0);
    int budget = 30000;
    std::vector<uint64_t> fw, bw;
    auto emit = [&]() {
      // sequence: el0 b0 el1 b1 ... elk ; direction canonical
      size_t L = path.size();
      fw.clear(); bw.clear();
      for (size_t k = 0; k < L; ++k) { fw.push_back(el[path[k]]); if (k + 1 < L) fw.push_back(100 + bt[k]); }
      bw.assign(fw.rbegin(), fw.rend());
      const auto& s = fw < bw ? fw : bw;
      out.push_back(tagk(seqHash(s), 6));
    };
    std::function<void(int)> dfs = [&](int u) {
      if (--budget < 0) return;
      if (path.size() >= 3) emit();
      if (path.size() >= 7) return;
      for (const auto& bnd : mol.atomBonds(mol.getAtomWithIdx(u))) {
        int v = int(bnd->getOtherAtomIdx(u));
        if (on[v]) continue;
        on[v] = 1; path.push_back(v);
        bt.push_back(bnd->getIsAromatic() ? 12 : uint64_t(bnd->getBondType()));
        dfs(v);
        bt.pop_back(); path.pop_back(); on[v] = 0;
      }
    };
    for (int s = 0; s < n && budget > 0; ++s) { on[s] = 1; path.assign(1, s); dfs(s); on[s] = 0; }
  }
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

}  // namespace fp3
