// Enumerate the fragments a candidate structure could break into, and build
// a predicted spectrum from their masses - the "forward model" idea from
// casmi/reference/forward-model-sho-saga.html, done without a learned GNN:
// rule-based intensities instead of ICEBERG's trained ones.
//
// v1 scope, deliberately narrow:
//   - break one non-ring bond at a time (the common case: losing a
//     substituent or side chain)
//   - break two bonds belonging to the same ring at a time (the only way a
//     ring bond can disconnect anything - cutting one ring bond alone never
//     separates the molecule, it just opens the ring)
//   - a cut's fragment mass is the sum of its atoms' own masses plus their
//     EXISTING implicit/explicit H count, read from the intact molecule -
//     no attempt yet at modelling the H-rearrangement a real cleavage
//     causes. That is exactly the kind of heuristic knob this project has
//     always tuned against a holdout rather than guessed (see sim_power,
//     the mass-prior sigma, the instrument bonus, ...); it needs its own
//     sweep once this compiles and runs, not a value invented here.
//   - every generated fragment gets equal weight; same reasoning.
//
// Build (from vcvars64):
//   cmake --build build   (see rdkit-min/CMakeLists.txt)
//
// Usage:
//   fragment_enum <smiles>
#include <algorithm>
#include <iostream>
#include <map>
#include <set>
#include <vector>

#include <GraphMol/GraphMol.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/PeriodicTable.h>
#include <GraphMol/RingInfo.h>
#include <GraphMol/SmilesParse/SmilesParse.h>

namespace {

double fragmentMass(const RDKit::ROMol& mol, const std::vector<int>& atomIdx) {
  const RDKit::PeriodicTable* pt = RDKit::PeriodicTable::getTable();
  double mass = 0.0;
  for (int idx : atomIdx) {
    const RDKit::Atom* a = mol.getAtomWithIdx(idx);
    mass += pt->getMostCommonIsotopeMass(a->getAtomicNum());
    mass += double(a->getTotalNumHs(/*includeNeighbors=*/false)) *
            pt->getMostCommonIsotopeMass(1);
  }
  return mass;
}

// All (atom-index-list) fragments produced by removing exactly the given
// bonds, or empty if that cut does not actually disconnect the molecule
// (e.g. one ring bond on its own, or two bonds that both sit on a path
// already reachable another way in a fused/bridged ring system).
std::vector<std::vector<int>> fragmentsAfterCutting(
    const RDKit::ROMol& mol, const std::vector<unsigned int>& bondIdx) {
  RDKit::RWMol rw(mol);
  // Remove by atom-index pair, not by bond index - indices shift as bonds
  // are removed one at a time.
  for (unsigned int bi : bondIdx) {
    const RDKit::Bond* b = mol.getBondWithIdx(bi);
    rw.removeBond(b->getBeginAtomIdx(), b->getEndAtomIdx());
  }
  std::vector<std::vector<int>> frags;
  RDKit::MolOps::getMolFrags(rw, frags);
  if (frags.size() < 2) return {};  // did not actually disconnect anything
  return frags;
}

}  // namespace

int main(int argc, char** argv) {
  std::string smi = argc > 1 ? argv[1] : "CC(=O)Oc1ccccc1C(=O)O";  // aspirin
  std::unique_ptr<RDKit::RWMol> mol(RDKit::SmilesToMol(smi));
  if (!mol) {
    std::cerr << "failed to parse: " << smi << "\n";
    return 1;
  }
  const RDKit::RingInfo* ri = mol->getRingInfo();

  // One predicted peak per DISTINCT fragment mass, intensity = how many
  // ways (bond cuts) produced a fragment of that mass - the closest thing
  // to "weight" this v1 has, before a real heuristic replaces it.
  std::map<long long, int> peaks;  // mass rounded to milli-Da -> count
  auto addPeak = [&](double mass) {
    peaks[llround(mass * 1000.0)] += 1;
  };

  // the intact molecule itself - always a candidate peak (the precursor,
  // or an in-source fragment that lost nothing)
  std::vector<int> allAtoms(mol->getNumAtoms());
  for (size_t i = 0; i < allAtoms.size(); ++i) allAtoms[i] = int(i);
  addPeak(fragmentMass(*mol, allAtoms));

  int nCuts = 0;
  // single non-ring bonds
  for (const auto& bond : mol->bonds()) {
    if (ri->numBondRings(bond->getIdx()) > 0) continue;  // ring bond, needs a partner
    auto frags = fragmentsAfterCutting(*mol, {unsigned(bond->getIdx())});
    if (frags.empty()) continue;
    ++nCuts;
    for (const auto& f : frags) addPeak(fragmentMass(*mol, f));
  }
  // pairs of bonds in the same ring
  for (unsigned int r = 0; r < ri->numRings(); ++r) {
    const auto& ringBonds = ri->bondRings()[r];
    for (size_t i = 0; i < ringBonds.size(); ++i) {
      for (size_t j = i + 1; j < ringBonds.size(); ++j) {
        auto frags = fragmentsAfterCutting(
            *mol, {unsigned(ringBonds[i]), unsigned(ringBonds[j])});
        if (frags.empty()) continue;
        ++nCuts;
        for (const auto& f : frags) addPeak(fragmentMass(*mol, f));
      }
    }
  }

  std::cout << "smiles: " << smi << "\n"
            << "cuts tried that disconnected something: " << nCuts << "\n"
            << "distinct predicted masses: " << peaks.size() << "\n";
  std::vector<std::pair<double, int>> sorted;
  for (auto& kv : peaks) sorted.emplace_back(kv.first / 1000.0, kv.second);
  std::sort(sorted.begin(), sorted.end());
  for (auto& p : sorted)
    std::cout << "  m/z " << p.first << "   weight " << p.second << "\n";
  return 0;
}
