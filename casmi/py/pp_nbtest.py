"""Offline test of the v28 notebook's merge + identity dedup (cell 7, after the ranker run).
usage: pp_nbtest.py <notebook.ipynb> <workdir with direct_submission.csv, pool_candidates.csv, sample_submission.csv>
Writes <workdir>/submission.csv. multiprocessing.Pool is swapped for a thread pool (Windows spawn cannot
pickle functions defined by exec); on Kaggle (Linux fork) the real Pool is used.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

nb = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
c7 = "".join(nb["cells"][7]["source"])
code = c7[c7.index("import csv"):]
code = code.replace("from multiprocessing import Pool as _Pool", "from multiprocessing.dummy import Pool as _Pool")
WORK = Path(sys.argv[2])
DATA = WORK
t0 = time.time()
direct = None
g = dict(WORK=WORK, DATA=DATA, t0=t0, time=time, sys=sys, subprocess=subprocess, Path=Path)
for name in ("tight_count.csv", "best_score.csv"):
    p = WORK / name
    if not p.exists():
        rows = [l.split(",")[0] for l in (WORK / "direct_submission.csv").read_text(encoding="utf-8").splitlines()[1:]]
        hdr = "molecule_id,tight_count\n" if name == "tight_count.csv" else "molecule_id,best_score,top1_correct\n"
        p.write_text(hdr + "".join(f"{r},0{',0' if 'best' in name else ''}\n" for r in rows), encoding="utf-8")
exec(compile(code, "cell7", "exec"), g)
