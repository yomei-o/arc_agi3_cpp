// Smallest possible program against the from-scratch minimal RDKit build
// (casmi/cpp/rdkit-min/) - just RDGeneral + DataStructs + Geometry +
// GraphMol + SmilesParse, no Descriptors/FileParsers/etc. Confirms the
// build chain works and that parsing + basic graph traversal behave as
// expected, before writing the fragment enumerator against it.
#include <iostream>
#include <GraphMol/GraphMol.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/RingInfo.h>

int main(int argc, char** argv) {
  std::string smi = argc > 1 ? argv[1] : "CC(=O)Oc1ccccc1C(=O)O";  // aspirin
  std::unique_ptr<RDKit::RWMol> mol(RDKit::SmilesToMol(smi));
  if (!mol) {
    std::cerr << "failed to parse: " << smi << "\n";
    return 1;
  }
  std::cout << "smiles: " << smi << "\n"
            << "atoms (heavy only, implicit H not counted): " << mol->getNumAtoms() << "\n"
            << "bonds: " << mol->getNumBonds() << "\n";

  const RDKit::RingInfo* ri = mol->getRingInfo();
  std::cout << "rings: " << ri->numRings() << "\n";

  for (const auto& bond : mol->bonds()) {
    bool in_ring = ri->numBondRings(bond->getIdx()) > 0;
    std::cout << "  bond " << bond->getIdx() << ": "
              << bond->getBeginAtomIdx() << "-" << bond->getEndAtomIdx()
              << (in_ring ? "  (ring)" : "") << "\n";
  }
  return 0;
}
