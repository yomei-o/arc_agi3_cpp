"""make_v33_nb: notebook/casmi_match_v33.ipynb from casmi_match_v30.ipynb (pc5_STATUS.md).

v33 = v30 + the two-population Boltzmann merge of the main list and the fused PubChem list (phys_v33pc
alloc=boltz) in place of the gated fixed slots + promotion. phys_v33pc.cpp (+ rev/fp3_v32.h, which it
includes) must be in the dataset's rdkit-min/; if phys_v33pc.cpp is missing the notebook runs as v30 and
prints BOLTZ DISABLED.
usage: python make_v33_nb.py   (from anywhere; paths are relative to this file)
"""
import json
from pathlib import Path

BOLTZ = ",alloc=boltz,bT1=2,bT2=1,bpi=0.2,ba=0.25"  # pc5_STATUS.md: the chosen setting

nbd = Path(__file__).resolve().parent.parent / "notebook"
nb = json.loads((nbd / "casmi_match_v30.ipynb").read_text(encoding="utf-8"))


def edit(ci, old, new):
    src = "".join(nb["cells"][ci]["source"])
    assert src.count(old) == 1, (ci, old)
    src = src.replace(old, new)
    nb["cells"][ci]["source"] = src.splitlines(keepends=True)


edit(6, '    srcs_fp = ["phys_v30pc.cpp"] + srcs[1:]',
     '    V33 = (RDKIT_MIN / "phys_v33pc.cpp").exists()  # v33: Boltzmann merge of main + PubChem lists (pc5_STATUS.md)\n'
     '    print("phys_v33pc.cpp present:", V33, "" if V33 else "- BOLTZ DISABLED, running as v30", flush=True)\n'
     '    srcs_fp = ["phys_v33pc.cpp" if V33 else "phys_v30pc.cpp"] + srcs[1:]')
edit(7, '        V27_OPT += ",pcrev=tan"',
     '        if globals().get("V33"):\n'
     '            V27_OPT += "' + BOLTZ + '"  # v33: p_i=(1-pi)softmax(s/T1), q_j=pi softmax(t/T2), union by probability\n'
     '        else:\n'
     '            print("BOLTZ DISABLED (phys_v33pc not built): v30 slots", flush=True)\n'
     '        V27_OPT += ",pcrev=tan"')
(nbd / "casmi_match_v33.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
meta = json.loads((nbd / "kernel-metadata-v30.json").read_text(encoding="utf-8"))
meta.update(id="yomeiotani/casmi26-cpp-library-search-v33", title="CASMI26 cpp library search v33",
            code_file="casmi_match_v33.ipynb")
(nbd / "kernel-metadata-v33.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
print("wrote", nbd / "casmi_match_v33.ipynb", "and kernel-metadata-v33.json; option", BOLTZ)

# v33nofp: the same without the public FPNet weights (host ruling 2026-10-21: no FPNet in the final two
# submissions). As casmi_match_v27c_noFPNet.ipynb: the FPNet stage still compiles phys_v33pc, FP_ARGS = ["", "0"].
src = "".join(nb["cells"][6]["source"])
a = src.index("    FPNET_SRC = RDKIT_MIN.parent")
b = src.index("    # phys_fp = phys_ana + optional")
block = "".join("    " + l if l.strip() else l for l in src[a:b].splitlines(keepends=True))
src = src[:a] + "    if USE_FPNET:\n" + block + src[b:]
old = '    FP_ARGS = ["fpnet_test.txt", "1.0"]\n'
assert src.count(old) == 1
src = src.replace(old, '    FP_ARGS = ["fpnet_test.txt", "1.0"] if USE_FPNET else ["", "0"]  # v33nofp: no FPNet term\n')
old = "FP_ARGS = []\ntry:\n"
assert src.count(old) == 1
src = src.replace(old, "# v33nofp: the public FPNet weights (no released training code) are not used (host ruling);\n"
                       "# the stage still compiles phys_v33pc. The holdout numbers of pc5_STATUS.md are FPNet-free too.\n"
                       "USE_FPNET = False\nFP_ARGS = []\ntry:\n", 1)
compile(src, "cell6", "exec")
nb["cells"][6]["source"] = src.splitlines(keepends=True)
(nbd / "casmi_match_v33nofp.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
meta.update(id="yomeiotani/casmi26-cpp-library-search-v33nofp", title="CASMI26 cpp library search v33nofp",
            code_file="casmi_match_v33nofp.ipynb")
(nbd / "kernel-metadata-v33nofp.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
print("wrote", nbd / "casmi_match_v33nofp.ipynb", "and kernel-metadata-v33nofp.json")
