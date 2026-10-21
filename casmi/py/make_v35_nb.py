"""make_v35_nb: notebook/casmi_match_v35nofp.ipynb from casmi_match_v33nofp_pb.ipynb (v35_STATUS.md).

v35 = phys_v35pc = v33 (two-population Boltzmann merge, alloc=boltz) + v34 (attention retrieval engine over the
library, attn=1, pcrank=rev, attpool) with the own fp3 reverse model (pcfp), FPNet-free. Needs in the datasets:
  casmi26-rdkit-min:     rdkit-min/phys_v35pc.cpp (+ rdkit-min/rev/fp3_v32.h, which it includes)
  casmi26-pool-expanded: pool730/rev3_all_fp3c16k_big.bin (549 MB, the production fp3 model, trained on all of train)
Fallbacks (printed): no phys_v35pc.cpp -> v33 path (phys_v33pc / v30, as before); no fp3 model -> v35 binary
without pcfp/attn and the v33 merge setting (= v33 output).
The prebuilt hook (cell 2) is kept; the build kernel links every phys_v*.cpp in the dataset, so it covers
phys_v35pc once it is re-run on the new dataset version.
usage: python make_v35_nb.py   (paths are relative to this file)
"""
import json
from pathlib import Path

BOLTZ33 = ",alloc=boltz,bT1=2,bT2=1,bpi=0.2,ba=0.25"  # pc5_STATUS.md (REV2-tan PubChem list scale)
# v35_STATUS.md: re-tuned on the attention score scale was bT1=2,bT2=0.5,bpi=0.2,ba=0.5; the LB probe
# (2026-10-21, 0.023) puts the out-of-pool share at ~2-4%, so the conservative setting (pi 0.1) is used.
BOLTZ35 = ",alloc=boltz,bT1=3,bT2=0.5,bpi=0.1,ba=0.5"
ATT35 = ",attn=1,attpool=2"  # attn_STATUS.md / v35_STATUS.md

nbd = Path(__file__).resolve().parent.parent / "notebook"
nb = json.loads((nbd / "casmi_match_v33nofp_pb.ipynb").read_text(encoding="utf-8"))


def edit(ci, old, new):
    src = "".join(nb["cells"][ci]["source"])
    assert src.count(old) == 1, (ci, old)
    src = src.replace(old, new)
    compile(src, "cell%d" % ci, "exec")
    nb["cells"][ci]["source"] = src.splitlines(keepends=True)


edit(7, '    srcs_fp = ["phys_v33pc.cpp" if V33 else "phys_v30pc.cpp"] + srcs[1:]',
     '    V35 = (RDKIT_MIN / "phys_v35pc.cpp").exists()  # v35: v33 merge + v34 attention engine (v35_STATUS.md)\n'
     '    print("phys_v35pc.cpp present:", V35, "" if V35 else "- ATTN DISABLED, running as v33", flush=True)\n'
     '    srcs_fp = ["phys_v35pc.cpp" if V35 else "phys_v33pc.cpp" if V33 else "phys_v30pc.cpp"] + srcs[1:]')
edit(8, '        if globals().get("V33"):\n'
        '            V27_OPT += "' + BOLTZ33 + '"',
     '        FP3 = POOL730_DIR / "rev3_all_fp3c16k_big.bin"  # v35: own fp3 reverse model (pc4_STATUS.md), 549 MB\n'
     '        ATT = bool(globals().get("V35")) and FP3.exists()\n'
     '        print("fp3 model:", FP3, FP3.exists(), flush=True)\n'
     '        if ATT:\n'
     '            # v35: attention retrieval engine (library softmax(sim/T) -> fp3 bits) in the PubChem window and the\n'
     '            # pool fusion; PubChem list ordered by its own score (pcrank=rev); merge re-tuned on the new scale\n'
     '            V27_OPT = V27_OPT.replace(",pcrank=fused", ",pcrank=rev")\n'
     '            V27_OPT += ",pcfp=" + str(FP3) + "' + ATT35 + BOLTZ35 + '"\n'
     '        elif globals().get("V33") or globals().get("V35"):\n'
     '            print("ATTN DISABLED (no phys_v35pc or no fp3 model): v33 merge", flush=True)\n'
     '            V27_OPT += "' + BOLTZ33 + '"')
out = nbd / "casmi_match_v35nofp.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
meta = json.loads((nbd / "kernel-metadata-v33nofp_pb.json").read_text(encoding="utf-8"))
meta.update(id="yomeiotani/casmi26-cpp-library-search-v35nofp", title="CASMI26 cpp library search v35nofp",
            code_file="casmi_match_v35nofp.ipynb")
(nbd / "kernel-metadata-v35nofp.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
print("wrote", out, "and kernel-metadata-v35nofp.json;", ATT35 + BOLTZ35)
