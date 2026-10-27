"""fw_parity.py: score parity of the own C++ fw_score (GLACIER + ICEBERG) with the public Python runners.

    python fw_parity.py <test.parquet> <pool_candidates.csv> <n molecules> <out dir> <ice pkg> <gl pkg> <site>
                        <fw_score.exe> <glacier_w.bin> <ice_gen_w.bin> <ice_inten_w.bin> [threads=8]
Input = same-formula groups of the pool list top 25 (as v41), positive [M+H]+ / [M+Na]+ spectra (the runners'
coverage; GLACIER keeps [M+H]+ only). ICEBERG roots are the runner's rm_stereo -> InChI round trip.
"""
import csv, os, subprocess, sys, time

import numpy as np
import pandas as pd

TEST, POOL, NMOL, OUT, ICEPKG, GLPKG, SITE, EXE, GLW, IGW, IIW = sys.argv[1:12]
NT = sys.argv[12] if len(sys.argv) > 12 else "8"
NMOL = int(NMOL)
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, ICEPKG); sys.path.insert(0, GLPKG)
import fuse, gl_fuse  # noqa: E402
import ice_runner as IR  # noqa: E402

a = IR.parse([ICEPKG, "x", "y", "--site", SITE, "--any-rdkit", "--device", "cpu"])
IR.setup_env(a, IR.Log(True))
import ms_pred.common as common  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem.rdMolDescriptors import CalcMolFormula  # noqa: E402

RDLogger.DisableLog("rdApp.*")
test = pd.read_parquet(TEST)
pool = {r[0]: r[1] for r in csv.reader(open(POOL, encoding="utf-8")) if r and r[0] != "molecule_id"}
pos = set(test[(test.ionization_mode == "positive") & test.adduct.isin(["[M+H]+", "[M+Na]+"])].molecule_id)
mids = [m for m in dict.fromkeys(test.molecule_id) if m in pos and m in pool][:NMOL]
cands = {}
for mid in mids:
    smis = [s for s in pool[mid].split(";") if s][:25]
    forms = []
    for s in smis:
        m = Chem.MolFromSmiles(s)
        forms.append(CalcMolFormula(m) if m is not None else "?" + s)
    c = fuse.ice_candidates(smis, forms, top_n=25)
    if c:
        cands[mid] = c
items = [it for it in fuse.build_ice_input(test, cands) if it["spectra"]]
gl_items = [dict(it, spectra=[sp for sp in it["spectra"] if sp.get("adduct") == "[M+H]+"]) for it in items]
gl_items = [it for it in gl_items if it["spectra"]]
root = {}
for it in items:
    for c in it["cands"]:
        if c not in root:
            try:
                root[c] = common.smiles_from_inchi(common.inchi_from_smiles(common.rm_stereo(c))) or ""
            except Exception:
                root[c] = ""
print("molecules", len(items), "candidates", sum(len(it["cands"]) for it in items), flush=True)


def fmt(v):
    return "" if v is None else str(v)


with open(os.path.join(OUT, "fw_in.txt"), "w", encoding="utf-8") as f:
    for it in items:
        f.write(f"M {it['mid']}\n")
        for c in it["cands"]:
            f.write(f"C {c}\t{root.get(c, '')}\n" if root.get(c) else f"C {c}\n")
        for sp in it["spectra"]:
            ce_ev = ",".join(repr(float(x)) for x in (sp.get("ce_ev") or []))
            peaks = ",".join(f"{m!r}:{i!r}" for m, i in zip(sp["mz"], sp["it"]))
            f.write("S " + "\t".join([fmt(sp.get("adduct")), fmt(sp.get("mode")), fmt(sp.get("instrument")),
                                      fmt(sp.get("ce_units")), fmt(sp.get("ce_orig")), ce_ev, repr(float(sp["prec"])),
                                      peaks]) + "\n")
t = time.time()
r = subprocess.run([EXE, os.path.join(OUT, "fw_in.txt"), os.path.join(OUT, "fw_cpp.tsv"), f"gl={GLW}",
                    f"ice={IGW},{IIW}", f"threads={NT}"], capture_output=True, text=True)
print(r.stderr.strip(), f"(C++ total {time.time() - t:.0f}s)", flush=True)
cpp = {}
for l in open(os.path.join(OUT, "fw_cpp.tsv"), encoding="utf-8"):
    m, s, g, i = l.rstrip("\n").split("\t")
    cpp[(m, s)] = (None if g == "nan" else float(g), None if i == "nan" else float(i))
t = time.time()
py_ice = fuse.run_ice(ICEPKG, items, workdir=os.path.join(OUT, "ice_py"), device="cuda", budget_s=3000, site=SITE,
                      extra_args=["--any-rdkit"], log=None)
t1 = time.time()
py_gl = gl_fuse.run_gl(GLPKG, gl_items, workdir=os.path.join(OUT, "gl_py"), device="cuda", budget_s=3000, site=SITE,
                       extra_args=["--any-rdkit"], log=None)
print(f"python runners: ice {t1 - t:.0f}s, gl {time.time() - t1:.0f}s", flush=True)
for name, py, col in (("iceberg", py_ice, 1), ("glacier", py_gl, 0)):
    a, b, nn, on = [], [], 0, 0
    for it in items:
        for c in it["cands"]:
            x, y = py.get(it["mid"], {}).get(c), cpp.get((it["mid"], c), (None, None))[col]
            if x is None and y is None:
                nn += 1
            elif x is None or y is None:
                on += 1
            else:
                a.append(x); b.append(y)
    a, b = np.array(a), np.array(b)
    agree = tot = 0
    for it in items:
        xs = [(py.get(it["mid"], {}).get(c), cpp.get((it["mid"], c), (None, None))[col]) for c in it["cands"]]
        xs = [p for p in xs if p[0] is not None and p[1] is not None]
        for i in range(len(xs)):
            for j in range(i + 1, len(xs)):
                if xs[i][0] != xs[j][0]:
                    tot += 1
                    agree += (xs[i][0] > xs[j][0]) == (xs[i][1] > xs[j][1])
    d = np.abs(a - b) if len(a) else np.zeros(1)
    print(f"{name}: pairs {len(a)} both-null {nn} one-null {on}  pearson {np.corrcoef(a, b)[0, 1] if len(a) > 1 else 0:.5f}"
          f"  mean|diff| {d.mean():.4f} max {d.max():.3f}  pairwise order {agree}/{tot} = {agree / max(1, tot):.4f}")
