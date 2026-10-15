# InChIKey14 cross-check of the PubChem picks (top-k of the PubChem list per query):
# the C++ truth flag (non-isomeric canonical SMILES) vs RDKit InChIKey first block == query name.
import sys
sys.path.insert(0, r"C:\prog\casmi\pcslot")
import pcslot_grid as G
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
d, tag, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
plain, Q = G.load(d, tag)
rk = G.ranker(w_pop=2)
both = flagOnly = ikOnly = bad = n = 0
for name, q in Q.items():
    for c in rk(q)[:k]:
        n += 1
        m = Chem.MolFromSmiles(c["smi"])
        if m is None: bad += 1; continue
        ik = Chem.MolToInchiKey(m)[:14]
        hit = ik == name
        both += hit and c["truth"]; flagOnly += c["truth"] and not hit; ikOnly += hit and not c["truth"]
print(f"{tag}: picks {n}, unparsable {bad}, truth by both {both}, flag only {flagOnly}, InChIKey14 only {ikOnly}")
