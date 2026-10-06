// The forward model: candidate structure -> predicted spectrum.
//
// Rule-based stand-in for ICEBERG's trained fragment-generator + intensity
// GNN (casmi/reference/forward-model-sho-saga.html). Fragments come from
// breaking one non-ring bond, or two bonds in the same ring (see
// fragment_enum.cpp for why - one ring bond alone never disconnects
// anything). A real cleavage also transfers 0-2 hydrogens across the break,
// which this models as extra peaks at mass +/- k*massH, weighted down by
// how many H's moved - a free parameter (h_shift_decay) meant to be swept
// against a holdout exactly like sim_power, the mass-prior sigma and the
// instrument bonus were, not guessed here.
#pragma once

#include <cmath>
#include <map>
#include <memory>
#include <vector>

#include <GraphMol/GraphMol.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/PeriodicTable.h>
#include <GraphMol/RingInfo.h>

namespace forward_model {

inline double fragmentMass(const RDKit::ROMol& mol,
                            const std::vector<int>& atomIdx) {
  const RDKit::PeriodicTable* pt = RDKit::PeriodicTable::getTable();
  double mass = 0.0;
  for (int idx : atomIdx) {
    const RDKit::Atom* a = mol.getAtomWithIdx(idx);
    mass += pt->getMostCommonIsotopeMass(a->getAtomicNum());
    mass += double(a->getTotalNumHs(false)) * pt->getMostCommonIsotopeMass(1);
  }
  return mass;
}

inline std::vector<std::vector<int>> fragmentsAfterCutting(
    const RDKit::ROMol& mol, const std::vector<unsigned int>& bondIdx) {
  RDKit::RWMol rw(mol);
  for (unsigned int bi : bondIdx) {
    const RDKit::Bond* b = mol.getBondWithIdx(bi);
    rw.removeBond(b->getBeginAtomIdx(), b->getEndAtomIdx());
  }
  std::vector<std::vector<int>> frags;
  RDKit::MolOps::getMolFrags(rw, frags);
  if (frags.size() < 2) return {};
  return frags;
}

struct Params {
  int maxHShift = 2;          // how many H's a cleavage may transfer, each way
  double hShiftDecay = 1.0;   // weight *= exp(-hShiftDecay * k^2) per shift k
  double massH = 1.007825;
};

// (m/z, intensity) pairs, not yet merged/sorted - caller's cosine/entropy
// code already walks an unsorted list fine (match.cpp's does), but sorting
// once here is cheap and makes the output readable standalone.
inline std::vector<std::pair<double, double>> predictSpectrum(
    const RDKit::ROMol& mol, const Params& p = Params()) {
  std::map<long long, double> peaks;  // mass rounded to milli-Da -> intensity
  auto addFragment = [&](double mass) {
    for (int k = -p.maxHShift; k <= p.maxHShift; ++k) {
      double shifted = mass + double(k) * p.massH;
      double w = std::exp(-p.hShiftDecay * double(k * k));
      peaks[llround(shifted * 1000.0)] += w;
    }
  };

  const RDKit::RingInfo* ri = mol.getRingInfo();

  std::vector<int> allAtoms(mol.getNumAtoms());
  for (size_t i = 0; i < allAtoms.size(); ++i) allAtoms[i] = int(i);
  addFragment(fragmentMass(mol, allAtoms));

  for (const auto& bond : mol.bonds()) {
    if (ri->numBondRings(bond->getIdx()) > 0) continue;
    auto frags = fragmentsAfterCutting(mol, {unsigned(bond->getIdx())});
    for (const auto& f : frags) addFragment(fragmentMass(mol, f));
  }
  for (unsigned int r = 0; r < ri->numRings(); ++r) {
    const auto& ringBonds = ri->bondRings()[r];
    for (size_t i = 0; i < ringBonds.size(); ++i) {
      for (size_t j = i + 1; j < ringBonds.size(); ++j) {
        auto frags = fragmentsAfterCutting(
            mol, {unsigned(ringBonds[i]), unsigned(ringBonds[j])});
        for (const auto& f : frags) addFragment(fragmentMass(mol, f));
      }
    }
  }

  std::vector<std::pair<double, double>> out;
  out.reserve(peaks.size());
  for (auto& kv : peaks) out.emplace_back(kv.first / 1000.0, kv.second);
  return out;
}

}  // namespace forward_model
