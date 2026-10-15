"""500 more natural products for the QM set: COCONUT-listed train molecules
(coconut_fit_keys) from flib, not in any holdout query set, <= 40 heavy atoms."""
import random
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
B = "C:/prog/casmi/"
keys = set(l.strip().split("\t")[0] for l in open(B + "coconut_fit_keys.txt"))
excl = set()
for q in ("cqry", "tqry", "fqry", "sqry", "dqry"):
    for l in open(B + q + "_mols.txt", encoding="utf-8"):
        excl.add(l.split("\t")[0])
cand = []
for l in open(B + "flib_mols.txt", encoding="utf-8"):
    a, _, s = l.rstrip("\r\n").partition("\t")
    if a in keys and a not in excl and s:
        cand.append((a, s))
print("candidates", len(cand))
random.seed(2026)
random.shuffle(cand)
out = []
for a, s in cand:
    m = Chem.MolFromSmiles(s)
    if m is None or not (8 <= m.GetNumHeavyAtoms() <= 40) or Chem.GetFormalCharge(m) != 0:
        continue
    out.append(f"{a}\t{s}\n")
    if len(out) >= 500:
        break
open(B + "qm/extra_mols.txt", "w", encoding="utf-8").writelines(out)
print("wrote", len(out))
