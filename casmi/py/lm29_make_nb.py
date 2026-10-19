"""lm29_make_nb.py - notebook/casmi_match.ipynb (v28a) -> notebook/casmi_match_v29lm.ipynb.

Changes (and nothing else):
  cell 6: srcs_fp = ["phys_v29lm.cpp"] + srcs[1:]   (phys_v29lm = phys_v28pc + v29 LambdaMART)
  cell 7: argv[25] = <POOL730_DIR>/lmart_v29_prod.txt when phys_v29lm was built (FP_ARGS) and the file exists
"""
import json
import sys
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebook"
src, dst = NB / "casmi_match.ipynb", NB / "casmi_match_v29lm.ipynb"
model = sys.argv[1] if len(sys.argv) > 1 else "lmart_v29_prod.txt"
nb = json.loads(src.read_text(encoding="utf-8"))


def edit(ci, old, new):
    s = "".join(nb["cells"][ci]["source"])
    if s.count(old) != 1:
        sys.exit(f"cell {ci}: anchor found {s.count(old)} times: {old[:60]!r}")
    s = s.replace(old, new)
    nb["cells"][ci]["source"] = s.splitlines(keepends=True)


edit(6, 'srcs_fp = ["phys_v28pc.cpp"] + srcs[1:]',
     'srcs_fp = ["phys_v29lm.cpp"] + srcs[1:]  # v29: phys_v29lm = phys_v28pc + LambdaMART ranker (argv[25])')
edit(7, 'print("v28 options:", V27_OPT, flush=True)\n',
     'print("v28 options:", V27_OPT, flush=True)\n'
     '# v29: LambdaMART ranker (lmart_v29, trained on the holdout2 sibling/xinstr + denovo + truth-removed sets,\n'
     '# gated: queries with a library match >= the model\'s gate keep the hand score). It replaces the hand\n'
     '# fusion\'s order on the hand fusion\'s per-query scale, so + 1.0 z(fpnet) and ppop act as in v28a.\n'
     'LM_ARGS = []\n'
     f'LM_MODEL = POOL730_DIR / "{model}"\n'
     'if FP_ARGS and LM_MODEL.exists():\n'
     '    LM_ARGS = [str(LM_MODEL)]\n'
     'print("v29 lmart:", LM_ARGS if LM_ARGS else "OFF (phys_v29lm not built or model missing)", flush=True)\n')
edit(7, '+ ["gen=1,cap=300,prior=gen2_prior_c.txt,gw=-2:0:-1:4:0:1.5,thr=4,grid=0" + V27_OPT],',
     '+ ["gen=1,cap=300,prior=gen2_prior_c.txt,gw=-2:0:-1:4:0:1.5,thr=4,grid=0" + V27_OPT] + LM_ARGS,')
for c in nb["cells"]:
    if c["cell_type"] == "code":
        c["outputs"] = []
        c["execution_count"] = None
dst.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("wrote", dst)
