"""make_v38_nb: notebook/casmi_match_v38nofp.ipynb from casmi_match_v37nofp.ipynb (tune38_STATUS.md).

v38nofp = v37nofp with the hand weights re-tuned jointly on the timsTOF holdouts (hxqry, hbqry, hbsqry) by
py/tune38_search.py. Only numbers change (no new C++: phys_v37pc as in v37): the positional fusion weights of the
ranker call (MLP, REV2, library^32, analog), the own-FPNet weight wf (FP_ARGS), the generation weights gw, and the
options attpool / cnnpool / ftpool / ppop / bT1. Every v37 guard, retry and fallback is unchanged.
usage: python make_v38_nb.py <tuned.json from tune38_search.py>
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tune38_search import opt_string  # noqa: E402

nbd = Path(__file__).resolve().parent.parent / "notebook"
nb = json.loads((nbd / "casmi_match_v37nofp.ipynb").read_text(encoding="utf-8"))
W = json.load(open(sys.argv[1]))["tuned"]
A = opt_string(W)
o = dict(kv.split("=", 1) for kv in A["opts"].split(","))
print("v38 weights:", A)


def edit(ci, old, new, n=1):
    src = "".join(nb["cells"][ci]["source"])
    assert src.count(old) == n, (ci, old, src.count(old))
    src = src.replace(old, new)
    compile(src, "cell%d" % ci, "exec")
    nb["cells"][ci]["source"] = src.splitlines(keepends=True)


edit(7, 'FP_ARGS = ["fpnet_test.txt", "1.0"] if FPNET_OK else ["", "0"]',
     'FP_ARGS = ["fpnet_test.txt", "%s"] if FPNET_OK else ["", "0"]  # v38: tuned (tune38_STATUS.md)' % A["wf"])
edit(7, '"+ 1.0 z(fpnet) (own weights)"', '"+ %s z(fpnet) (own weights)"' % A["wf"])
edit(8, '"fwd_mlp_rank_v2.bin", "0.25", "smiles",', '"fwd_mlp_rank_v2.bin", "%s", "smiles",' % A["mlp"])
edit(8, '"rev2_all_v4k.bin", "1.0", "train", "8", "32", "24",',
     '"rev2_all_v4k.bin", "%s", "train", "%s", "32", "%s",' % (A["rev"], A["lib"], A["ana"]))
edit(8, 'gw=-2:0:-1:4:0:1.5,thr=4', 'gw=%s,thr=4' % o["gw"])
edit(8, '",ppop=0.15,ppgate=0.9,ppopf="', '",ppop=%s,ppgate=0.9,ppopf="' % o["ppop"])
edit(8, ',attn=1,attpool=2,alloc=boltz,bT1=3,bT2=0.5,bpi=0.1,ba=0.5"',
     ',attn=1,attpool=%s,alloc=boltz,bT1=%s,bT2=0.5,bpi=0.1,ba=0.5"' % (o["attpool"], o["bT1"]))
edit(8, '",cnnpool=2,cnninst=0"', '",cnnpool=%s,cnninst=0"' % o["cnnpool"])
edit(8, '",ftpool=2"', '",ftpool=%s"' % o["ftpool"])
edit(8, '",ftpool=2,cnninst=0"', '",ftpool=%s,cnninst=0"' % o["ftpool"])
src0 = "".join(nb["cells"][0]["source"]) if nb["cells"][0]["cell_type"] == "markdown" else None

out = nbd / "casmi_match_v38nofp.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
meta = json.loads((nbd / "kernel-metadata-v37nofp.json").read_text(encoding="utf-8"))
meta.update(id="yomeiotani/casmi26-cpp-library-search-v38nofp", title="CASMI26 cpp library search v38nofp",
            code_file="casmi_match_v38nofp.ipynb")
(nbd / "kernel-metadata-v38nofp.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
for c in nb["cells"]:
    if c["cell_type"] == "code":
        compile("".join(c["source"]), "cell", "exec")
print("wrote", out, "and kernel-metadata-v38nofp.json")
