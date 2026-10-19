"""lm29_cv.py - grouped 5-fold CV of lmart_v29 over the holdout2 dumps (build machine).

usage: lm29_cv.py <tag> <mix> <drop> [extra lmart keys...]
  runs the 5 folds in parallel (thr=2 each = 10 threads), OOF scores -> C:\\prog\\casmi\\lm29\\cv_<tag>_<k>.txt,
  logs cv_<tag>_<k>.log. Folds group by molecule name across all sets (rm variants share their molecule's fold).
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"C:\prog\casmi")
EXE = r"C:\prog\casmi\fwdmodel\lm29\lmart_v29.exe"
SETS = [("has0", "S"), ("hbs0", "S"), ("hx0", "S"), ("ha0", "D"), ("hc0", "D"), ("hb0", "D"),
        ("ha1", "R"), ("hc1", "R"), ("hb1", "R")]
BASE_DROP = "q_nspec,q_mass,q_pol,q_adduct,q_instr,raw_phys,raw_rev"


def main():
    tag, mix, drop = sys.argv[1], sys.argv[2], sys.argv[3]
    extra = sys.argv[4:]
    drop = BASE_DROP + ("," + drop if drop and drop != "-" else "")
    train = ",".join(f"lm29\\{s}_f.txt:{g}" for s, g in SETS)
    procs = []
    for k in range(5):
        oof = ROOT / "lm29" / f"cv_{tag}_{k}.txt"
        if oof.exists():
            oof.unlink()
        cmd = [EXE, "train", f"train={train}", f"mix={mix}", f"drop={drop}", f"fold={k}/5", "valfrac=0.2", "es=150",
               "trees=1500", "metric=mrr", "thr=2", f"out=lm29\\cvm_{tag}_{k}.txt", f"oofs=lm29\\cv_{tag}_{k}.txt"] + extra
        log = open(ROOT / "lm29" / f"cv_{tag}_{k}.log", "w")
        procs.append(subprocess.Popen(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT))
    rc = [p.wait() for p in procs]
    print(tag, "rc", rc, flush=True)


if __name__ == "__main__":
    main()
