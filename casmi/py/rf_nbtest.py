"""End-to-end check of the v29rf notebook post-step on holdout2 outputs, in the production format.

rf_nbtest.py <notebook> <rfdir> <casmi dir>
Takes cell 7's rf_* functions out of the notebook (exec of the definitions only), turns the holdout main
lists (names) into SMILES through pool730_mols.txt (as pool_candidates.csv carries them in production),
feeds the library list (SMILES, match.cpp) and best_score.csv, and scores the merged list after mapping
it back to InChIKey14. Prints MRR@25 per component for RF_MODE off-equivalent (gate closed) and as set.
"""
import ast, csv, json, sys
from pathlib import Path

nb, RF, CAS = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
src = "".join(json.load(open(nb, encoding="utf-8"))["cells"][7]["source"])
tree = ast.parse(src)
keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("rf_")
        or isinstance(n, ast.Assign) and any(getattr(x, "id", "").startswith("RF_")
                                             for t in n.targets for x in (t.elts if isinstance(t, ast.Tuple) else [t]))]
ns = {"csv": csv}
exec(compile(ast.Module(body=keep, type_ignores=[]), "cell7", "exec"), ns)
print("notebook settings: mode", ns["RF_MODE"], "gate", ns["RF_GATE"])

name2smi = {}
for line in open(CAS / "pool730_mols.txt", encoding="utf-8"):
    t = line.rstrip("\n").split("\t")
    if len(t) >= 2 and t[0] not in name2smi:
        name2smi[t[0]] = t[1]
COMP = [("has0", "hasqry", "ylib"), ("hc0", "hcqry", "ylib"), ("hc1", "hcqry", "ylib"), ("hx0", "hxqry", "zlib"),
        ("ha0", "haqry", "ylib"), ("hb0", "hbqry", "ylib"), ("hbs0", "hbsqry", "ylib")]
s2k_cache = {}
for tag, q, lib in COMP:
    if not (RF / f"{tag}.csv").exists() or not (RF / f"lib_{q}.csv").exists():
        continue
    if lib not in s2k_cache:
        s2k_cache[lib] = ns["rf_smi2ik"]([CAS / "pool730_mols.txt", CAS / f"{lib}_mols.txt"])
    s2k = s2k_cache[lib]
    main = ns["rf_read"](RF / f"{tag}.csv")
    libl = ns["rf_read"](RF / f"lib_{q}.csv")
    best = {}
    with open(RF / f"lib_{q}_best.csv", encoding="utf-8") as f:
        r = csv.reader(f); next(r)
        for row in r:
            best[row[0]] = float(row[1])
    truth = [l.split("\t")[0] for l in open(CAS / f"{q}_mols.txt", encoding="utf-8") if l.strip()]
    res = {}
    for label, force in (("gate closed", False), ("as set", None)):
        tot = 0.0; nopen = 0
        for t in truth:
            a = [name2smi.get(s, s) for s in main.get(t, [])[:25]]  # production depth 25
            b = libl.get(t, [])[:25]
            g = (ns["RF_GATE"] is None or best.get(t, -1e9) >= ns["RF_GATE"]) if force is None else False
            nopen += int(g)
            m = ns["rf_merge"](a, b, s2k, g)
            names = [ns["rf_key"](s, s2k) for s in m]
            if t in names[:25]:
                tot += 1.0 / (names.index(t) + 1)
        res[label] = (tot / len(truth), nopen)
    print(f"{tag:5s} gate closed {res['gate closed'][0]:.4f}   as set {res['as set'][0]:.4f}   "
          f"diff {res['as set'][0]-res['gate closed'][0]:+.4f}   gate open {res['as set'][1]}/{len(truth)}")
