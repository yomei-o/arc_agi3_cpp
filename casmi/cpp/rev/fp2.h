// fp2: collision-free "vocabulary" circular fingerprints for the reverse
// model v2 (rev2_*). Instead of folding every environment hash into 2048
// bits (ecfp.h: unrelated substructures share a bit, so the network must
// predict an OR of them and a candidate gets credit for the wrong one),
// each molecule yields its set of 64-bit environment hashes - ECFP-style
// atom invariants up to radius rE and FCFP-style (pharmacophore class)
// invariants up to radius rF - and a fixed vocabulary of the most frequent
// hashes in the training library defines the bits. Hashes not in the
// vocabulary are simply not scored (they are rare in the library, so the
// model could not have learned them anyway).
//
// The ECFP environments use ecfp.h's invariants and update rule; every hash
// is re-tagged by (kind, radius) so ECFP and FCFP environments never collide.
#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <GraphMol/GraphMol.h>
#include <GraphMol/RingInfo.h>

namespace fp2 {

inline uint64_t mix(uint64_t h, uint64_t v) {
  h ^= v + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
  h *= 0xBF58476D1CE4E5B9ull;
  return h ^ (h >> 31);
}

// FCFP-like atom class bits: donor, acceptor, aromatic, halogen, basic, acidic.
inline uint64_t pharmClass(const RDKit::ROMol& mol, const RDKit::Atom* a) {
  int z = a->getAtomicNum();
  int nh = int(a->getTotalNumHs());
  int chg = a->getFormalCharge();
  bool arom = a->getIsAromatic();
  uint64_t c = 0;
  if ((z == 7 || z == 8) && nh > 0) c |= 1;                       // donor
  if (z == 8 && chg <= 0) c |= 2;                                 // acceptor (O)
  if (z == 7 && chg <= 0 && nh == 0) {
    // pyridine-like aromatic N (2 connections) or non-amide tertiary/imine N
    if (arom ? a->getDegree() == 2 : true) c |= 2;
  }
  if (arom) c |= 4;
  if (z == 9 || z == 17 || z == 35 || z == 53) c |= 8;
  if (z == 7 && !arom && chg >= 0) {
    // basic: sp3 amine N not attached to carbonyl/sulfonyl/aromatic atoms
    bool basic = true;
    for (const auto& bnd : mol.atomBonds(a)) {
      if (bnd->getBondType() != RDKit::Bond::SINGLE) { basic = false; break; }
      const RDKit::Atom* o = bnd->getOtherAtom(a);
      if (o->getIsAromatic()) { basic = false; break; }
      for (const auto& b2 : mol.atomBonds(o))
        if (b2->getBondType() == RDKit::Bond::DOUBLE) {
          int z2 = b2->getOtherAtom(o)->getAtomicNum();
          if (z2 == 8 || z2 == 16 || z2 == 7) { basic = false; break; }
        }
      if (!basic) break;
    }
    if (basic) c |= 16;
  }
  if (z == 8 && nh > 0 && a->getDegree() == 1) {
    // acidic OH: on C/S/P that also carries a double-bonded O
    const RDKit::Atom* o = nullptr;
    for (const auto& bnd : mol.atomBonds(a)) o = bnd->getOtherAtom(a);
    if (o && (o->getAtomicNum() == 6 || o->getAtomicNum() == 15 || o->getAtomicNum() == 16))
      for (const auto& b2 : mol.atomBonds(o))
        if (b2->getBondType() == RDKit::Bond::DOUBLE && b2->getOtherAtom(o)->getAtomicNum() == 8) { c |= 32; break; }
  }
  if (chg < 0) c |= 32;
  return c;
}

// Sorted unique environment hashes. rE / rF: max radius of the ECFP / FCFP
// kind (-1 = kind off).
inline std::vector<uint64_t> envs(const RDKit::ROMol& mol, int rE, int rF) {
  const int n = int(mol.getNumAtoms());
  const RDKit::RingInfo* ri = mol.getRingInfo();
  std::vector<uint64_t> out;
  std::vector<uint64_t> cur(n), nxt(n);
  std::vector<std::pair<uint64_t, uint64_t>> nb;
  for (int kind = 0; kind < 2; ++kind) {
    int R = kind == 0 ? rE : rF;
    if (R < 0) continue;
    for (int i = 0; i < n; ++i) {
      const RDKit::Atom* a = mol.getAtomWithIdx(i);
      uint64_t h;
      if (kind == 0) {
        h = 0x1234567ull;
        h = mix(h, uint64_t(a->getAtomicNum()));
        h = mix(h, uint64_t(a->getDegree()));
        h = mix(h, uint64_t(a->getTotalNumHs()));
        h = mix(h, uint64_t(int64_t(a->getFormalCharge()) + 8));
        h = mix(h, uint64_t(ri->numAtomRings(i) > 0));
        h = mix(h, uint64_t(a->getIsAromatic()));
      } else {
        h = mix(0x7654321ull, pharmClass(mol, a));
      }
      cur[i] = h;
      out.push_back(mix(mix(h, 0xABCDull + uint64_t(kind)), 0));
    }
    for (int r = 1; r <= R; ++r) {
      for (int i = 0; i < n; ++i) {
        nb.clear();
        for (const auto& bnd : mol.atomBonds(mol.getAtomWithIdx(i))) {
          int j = int(bnd->getOtherAtomIdx(i));
          nb.emplace_back(uint64_t(bnd->getBondType()), cur[j]);
        }
        std::sort(nb.begin(), nb.end());
        uint64_t h = mix(uint64_t(r), cur[i]);
        for (auto& p : nb) { h = mix(h, p.first); h = mix(h, p.second); }
        nxt[i] = h;
        out.push_back(mix(mix(h, 0xABCDull + uint64_t(kind)), uint64_t(r)));
      }
      cur.swap(nxt);
    }
  }
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

// Vocabulary: hash -> bit index.
struct Vocab {
  int rE = 3, rF = 3;
  std::vector<uint64_t> hashes;
  std::unordered_map<uint64_t, int> index;
  void build() {
    index.clear();
    for (size_t i = 0; i < hashes.size(); ++i) index.emplace(hashes[i], int(i));
  }
  std::vector<uint16_t> bits(const RDKit::ROMol& mol) const {
    std::vector<uint16_t> b;
    for (uint64_t h : envs(mol, rE, rF)) {
      auto it = index.find(h);
      if (it != index.end()) b.push_back(uint16_t(it->second));
    }
    std::sort(b.begin(), b.end());
    return b;
  }
};

}  // namespace fp2
