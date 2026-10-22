"""e2e_rerank: re-run only the ranker of a finished e2e notebook run with modified options (ablations), then the
notebook's merge (pool list first, library list fills, 25, 'CCO' if empty) -> <tag>_submission.csv in the work dir.
    python e2e_rerank.py <work dir> <tag> <nofpnet|nocnn|...> [threads-mask hex]
Options are rebuilt exactly as cell 8 of casmi_match_v36nofp.ipynb printed them ("v28 options:" in the log is
re-created here from the same input tree).
"""
import csv, ctypes, os, subprocess, sys, time
from pathlib import Path

W = Path(sys.argv[1]); tag = sys.argv[2]; mode = sys.argv[3]
ctypes.windll.kernel32.SetProcessAffinityMask(ctypes.windll.kernel32.GetCurrentProcess(),
                                              int(sys.argv[4] if len(sys.argv) > 4 else "F0", 16))
INP = Path(r"C:\prog\casmi\e2e\input"); P7 = INP / "rdkitmin" / "pool730"
opt = (",libcap=1,ppop=0.15,ppgate=0.9,ppopf=" + str(P7 / "pool730_pop.bin") + ",pc=1,pcv=2,pcrank=rev,relhi=6,relp=3,sp=6,"
       "pcdir=" + str(INP / "pubchem") + ",pcfp=" + str(P7 / "rev3_all_fp3c16k_big.bin")
       + ",attn=1,attpool=2,alloc=boltz,bT1=3,bT2=0.5,bpi=0.1,ba=0.5")
cnn = ",cnnmodel=" + str(P7 / "cnn_prod_c.bin") + ",cnnpool=2,cnninst=0"
fp = ["fpnet_test.txt", "1.0"]
if mode == "nofpnet":
    fp = ["", "0"]; opt += cnn
elif mode == "nocnn":
    pass
else:
    raise SystemExit("mode?")
opt += ",pcrev=tan"
out = tag + "_pool_candidates.csv"
t = time.time()
r = subprocess.run([str(W / "phys_rank.exe"), "eval", str(W), "test", "pool730", "10", "phys_size_fqry.txt", out,
                    "fwd_mlp_rank_v2.bin", "0.25", "smiles", "rev2_all_v4k.bin", "1.0", "train", "8", "32", "24", "",
                    "4", "3", "8000", ""] + fp + ["gen=1,cap=300,prior=gen2_prior_c.txt,gw=-2:0:-1:4:0:1.5,thr=4,grid=0" + opt],
                   capture_output=True, text=True)
(W / (tag + "_stdout.txt")).write_text(r.stdout + "\n" + r.stderr[-3000:], encoding="utf-8")
print(tag, "rc", r.returncode, "in %.0fs" % (time.time() - t), flush=True)


def rd(p):
    with open(p, encoding="utf-8") as f:
        return {row[0]: row[1] for row in csv.reader(f) if row and row[0] != "molecule_id"}


direct = rd(W / "direct_submission.csv"); pool = rd(W / out)
ids = [row[0] for row in csv.reader(open(INP / "comp" / "sample_submission.csv", encoding="utf-8"))][1:]
with open(W / (tag + "_submission.csv"), "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f); w.writerow(["molecule_id", "smiles"])
    for m in ids:
        merged, seen = [], set()
        for s in [s for s in pool.get(m, "").split(";") if s] + [s for s in direct.get(m, "").split(";") if s]:
            if s not in seen:
                seen.add(s); merged.append(s)
            if len(merged) == 25:
                break
        w.writerow([m, ";".join(merged) if merged else "CCO"])
print("wrote", W / (tag + "_submission.csv"))
