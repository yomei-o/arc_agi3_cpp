// ECFP4-style circular fingerprint on top of rdkit-min's GraphMol - the
// reverse model's prediction target (spectrum -> which substructures are
// present). RDKit's own Morgan implementation lives in GraphMol/Fingerprints,
// which rdkit-min never extracted; this is the same algorithm (iterated
// neighbourhood hashing, folded into nBits) written directly against the
// graph API rather than pulling in another module. It does NOT reproduce
// RDKit's exact bit assignments (different hash, no redundant-environment
// pruning) - it only has to be self-consistent, since the same function
// produces both the training targets and every candidate's fingerprint.
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include <GraphMol/GraphMol.h>
#include <GraphMol/RingInfo.h>

namespace ecfp {

inline uint64_t mix(uint64_t h, uint64_t v) {
  h ^= v + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
  h *= 0xBF58476D1CE4E5B9ull;
  return h ^ (h >> 31);
}

// Sorted, de-duplicated on-bit indices in [0, nBits).
inline std::vector<uint16_t> compute(const RDKit::ROMol& mol, int radius = 2, int nBits = 2048) {
  const int n = int(mol.getNumAtoms());
  const RDKit::RingInfo* ri = mol.getRingInfo();
  std::vector<uint64_t> cur(n), nxt(n);
  std::vector<uint16_t> bits;
  for (int i = 0; i < n; ++i) {
    const RDKit::Atom* a = mol.getAtomWithIdx(i);
    uint64_t h = 0x1234567ull;
    h = mix(h, uint64_t(a->getAtomicNum()));
    h = mix(h, uint64_t(a->getDegree()));
    h = mix(h, uint64_t(a->getTotalNumHs()));
    h = mix(h, uint64_t(int64_t(a->getFormalCharge()) + 8));
    h = mix(h, uint64_t(ri->numAtomRings(i) > 0));
    h = mix(h, uint64_t(a->getIsAromatic()));
    cur[i] = h;
    bits.push_back(uint16_t(h % uint64_t(nBits)));
  }
  std::vector<std::pair<uint64_t, uint64_t>> nb;
  for (int r = 1; r <= radius; ++r) {
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
      bits.push_back(uint16_t(h % uint64_t(nBits)));
    }
    cur.swap(nxt);
  }
  std::sort(bits.begin(), bits.end());
  bits.erase(std::unique(bits.begin(), bits.end()), bits.end());
  return bits;
}

}  // namespace ecfp
