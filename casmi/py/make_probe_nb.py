"""make_probe_nb.py: casmi/notebook/casmi_match.ipynb (v28a) -> casmi_match_probe.ipynb (LB probe:
submission = non-pool PubChem candidates only, phys_probe.cpp probe=1). See probe_STATUS.md."""
import json, ast, sys
from pathlib import Path

NB = Path(__file__).resolve().parent.parent / "notebook"
nb = json.loads((NB / "casmi_match.ipynb").read_text(encoding="utf-8"))
src = ["".join(c["source"]) for c in nb["cells"]]


def rep(i, old, new):
    assert src[i].count(old) == 1, (i, old[:60], src[i].count(old))
    src[i] = src[i].replace(old, new)


# cell 6: compile phys_probe.cpp (phys_v28pc + probe=1) in the FPNet stage
rep(6, 'srcs_fp = ["phys_v28pc.cpp"] + srcs[1:]',
    'srcs_fp = ["phys_probe.cpp"] + srcs[1:]  # PROBE: phys_probe = phys_v28pc + probe=1 (non-pool PubChem list only)')

# cell 7: PubChem tables + probe option; fallback build when the FPNet stage failed
rep(7, 'print("v28 options:", V27_OPT, flush=True)\n',
    '''# PROBE: the submission is ONLY PubChem window candidates that are not pool730 molecules (stereo-blind
# canonical SMILES vs the pool window), in the PubChem v2 order, top 25, every query (no gate).
# LB of this = (fraction of hidden answers outside the pool) x (MRR of this list on those).
if not FP_ARGS:
    # FPNet stage failed -> phys_probe was not built there; build it here (FPNet does not affect the probe list)
    t4 = time.time()
    r = subprocess.run(["g++", "-O3", "-std=c++20", "-pthread", "-w",
                        "-I", str(RDKIT_MIN), "-I", str(RDKIT_MIN / "boost-min"),
                        "-I", str(AUTOGRAD_DIR.parent),
                        "-o", str(WORK / "phys_probe")] + ["phys_probe.cpp"] + srcs[1:],
                       capture_output=True, text=True, cwd=str(RDKIT_MIN))
    print("PROBE fallback phys_probe compile rc:", r.returncode, "in %.0fs" % (time.time() - t4), flush=True)
    if r.returncode != 0:
        print(r.stderr[-4000:]); raise SystemExit("PROBE FAILED: phys_probe compile")
    os.replace(WORK / "phys_probe", WORK / "phys_rank")
    PHYS_PARAMS = "phys_size_fqry.txt"
    V27_OPT = ",libcap=1"
PC_DIR = None
for pat in ("*/pc_mass.bin", "*/*/pc_mass.bin", "*/*/*/pc_mass.bin", "*/*/*/*/pc_mass.bin"):
    hit = sorted(Path("/kaggle/input").glob(pat))
    if hit:
        PC_DIR = hit[0].parent; break
print("pubchem tables:", PC_DIR, flush=True)
if PC_DIR is None:
    raise SystemExit("PROBE FAILED: pc_mass.bin not found")
V27_OPT += ",pc=1,pcv=2,pcrank=fused,probe=1,pcout=plain_candidates.csv,pcdir=" + str(PC_DIR)
print("v28 options:", V27_OPT, flush=True)
''')
rep(7, 'print(r.stdout[-2000:], flush=True)\n',
    'print(r.stdout[-2000:], flush=True)\nfor _l in r.stdout.splitlines():\n    if _l.startswith("PROBE"): print("C++", _l, flush=True)\n')

# final list: the probe list only (no merge with the library / pool list)
i0 = src[7].index("# v13 merge:")
i1 = src[7].index("import pandas as pd")
src[7] = src[7][:i0] + '''# PROBE: no merge - the submission is the non-pool PubChem list as written by phys_probe (out.csv),
# already padded only with further non-pool PubChem candidates (pcn=50 >= 25).
merged_dict, lens = {}, []
for mol_id in sorted(set(direct) | set(pool_c)):
    p_list = [s for s in pool_c.get(mol_id, "").split(";") if s][:25]
    lens.append(len(p_list))
    merged_dict[mol_id] = ";".join(p_list) if p_list else "CCO"
import numpy as np
lens = np.array(lens)
print("PROBE submission: %d queries, average list length %.2f, full (25) %d, empty %d"
      % (len(lens), lens.mean(), int((lens == 25).sum()), int((lens == 0).sum())), flush=True)

''' + src[7][i1:]

for i, s in enumerate(src):
    if nb["cells"][i]["cell_type"] != "code":
        continue
    if i == 0 and s.lstrip().startswith("%%"):
        continue
    try:
        ast.parse(s)
    except SyntaxError as e:
        # cell 0 may be a %%writefile cell
        first = s.splitlines()[0] if s else ""
        if first.startswith("%%"):
            continue
        print("SYNTAX", i, e); sys.exit(1)
    nb["cells"][i]["source"] = s.splitlines(keepends=True)
    nb["cells"][i]["outputs"] = []
    nb["cells"][i]["execution_count"] = None
(NB / "casmi_match_probe.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", NB / "casmi_match_probe.ipynb")
