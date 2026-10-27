"""gl_parity.py: score parity of the own C++ gl_score against the public Python GLACIER runner on the same input.

    python gl_parity.py <test.parquet> <pool_candidates.csv> <n molecules> <out dir> <glacier pkg> <site> <gl_score.exe>
                        <glacier_w.bin>
Builds the runner input like the v41 notebook (same-formula groups of the pool list top 25, [M+H]+ spectra only),
writes the C++ text input, runs both, and prints per-candidate agreement (Pearson, max |diff|, same within-group order).
"""
import csv, json, os, subprocess, sys, time

import numpy as np
import pandas as pd

TEST, POOL, NMOL, OUT, PKG, SITE, EXE, WB = sys.argv[1:9]
NMOL = int(NMOL)
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(os.path.dirname(PKG), "casmi26-iceberg")); sys.path.insert(0, PKG)
import fuse, gl_fuse  # noqa: E402
from rdkit import Chem  # noqa: E402
from rdkit.Chem.rdMolDescriptors import CalcMolFormula  # noqa: E402

test = pd.read_parquet(TEST)
pool = {r[0]: r[1] for r in csv.reader(open(POOL, encoding="utf-8")) if r and r[0] != "molecule_id"}
mh = set(test[(test.ionization_mode == "positive") & (test.adduct == "[M+H]+")].molecule_id)
mids = [m for m in dict.fromkeys(test.molecule_id) if m in mh and m in pool][:NMOL]
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
items = fuse.build_ice_input(test, cands)
items = [dict(it, spectra=[sp for sp in it["spectra"] if sp.get("adduct") == "[M+H]+"]) for it in items]
items = [it for it in items if it["spectra"]]
print("molecules", len(items), "candidates", sum(len(it["cands"]) for it in items), "spectra",
      sum(len(it["spectra"]) for it in items), flush=True)


def fmt(v):
    return "" if v is None else str(v)


with open(os.path.join(OUT, "gl_in.txt"), "w", encoding="utf-8") as f:
    for it in items:
        f.write(f"M {it['mid']}\n")
        for c in it["cands"]:
            f.write(f"C {c}\n")
        for sp in it["spectra"]:
            ce_ev = ",".join(repr(float(x)) for x in (sp.get("ce_ev") or []))
            peaks = ",".join(f"{m!r}:{i!r}" for m, i in zip(sp["mz"], sp["it"]))
            f.write("S " + "\t".join([fmt(sp.get("adduct")), fmt(sp.get("mode")), fmt(sp.get("instrument")),
                                      fmt(sp.get("ce_units")), fmt(sp.get("ce_orig")), ce_ev, repr(float(sp["prec"])),
                                      peaks]) + "\n")
t = time.time()
r = subprocess.run([EXE, WB, os.path.join(OUT, "gl_in.txt"), os.path.join(OUT, "gl_cpp.tsv"), "8", "0", "1"],
                   capture_output=True, text=True)
print(r.stderr.strip(), f"(C++ {time.time() - t:.0f}s)", flush=True)
t = time.time()
py = gl_fuse.run_gl(PKG, items, workdir=os.path.join(OUT, "gl_py"), device="cuda", budget_s=3000, site=SITE,
                    extra_args=["--any-rdkit"], log=None)
print(f"python runner {time.time() - t:.0f}s", flush=True)
cpp = {}
for l in open(os.path.join(OUT, "gl_cpp.tsv"), encoding="utf-8"):
    m, s, v = l.rstrip("\n").split("\t")
    cpp[(m, s)] = None if v == "nan" else float(v)
a, b, both_none, one_none = [], [], 0, 0
for it in items:
    for c in it["cands"]:
        x, y = py.get(it["mid"], {}).get(c), cpp.get((it["mid"], c))
        if x is None and y is None:
            both_none += 1
        elif x is None or y is None:
            one_none += 1
        else:
            a.append(x); b.append(y)
a, b = np.array(a), np.array(b)
print(f"pairs {len(a)}  both null {both_none}  one null {one_none}")
if len(a):
    d = np.abs(a - b)
    print(f"pearson {np.corrcoef(a, b)[0, 1]:.6f}  max|diff| {d.max():.4g}  mean|diff| {d.mean():.3g}  "
          f"share |diff|<1e-3 {np.mean(d < 1e-3):.3f}  <1e-2 {np.mean(d < 1e-2):.3f}")
# within-molecule ranking agreement of the GL scores (what the re-ranker uses)
agree = tot = 0
for it in items:
    xs = [(py.get(it["mid"], {}).get(c), cpp.get((it["mid"], c))) for c in it["cands"]]
    xs = [p for p in xs if p[0] is not None and p[1] is not None]
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            if xs[i][0] == xs[j][0]:
                continue
            tot += 1
            agree += (xs[i][0] > xs[j][0]) == (xs[i][1] > xs[j][1])
print(f"pairwise order agreement {agree}/{tot} = {agree / max(1, tot):.4f}")
