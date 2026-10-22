"""v36_timeout_test: the FPNet-stage time guard of casmi_match_v36nofp.ipynb (cell 7 helpers) on Linux.
Runs the helper definitions taken verbatim from the notebook, then:
  1. a child that forks grandchildren (sh -c 'sleep & sleep & wait') under a 13 s budget -> TimeoutError and no
     leftover 'sleep 307' process (the grandchildren are killed through their new session);
  2. a command that finishes in time -> normal CompletedProcess;
  3. the chunked copy with an expired deadline -> TimeoutError.
usage (WSL / Linux): python3 v36_timeout_test.py <casmi_match_v36nofp.ipynb>
"""
import json, os, subprocess, sys, time
from pathlib import Path

nb = json.load(open(sys.argv[1], encoding="utf-8"))
src = "".join(nb["cells"][7]["source"])
helpers = src[:src.index("t3 = time.time()")]
g = {"os": os, "subprocess": subprocess, "time": time, "Path": Path}
exec(helpers, g)

t = time.time()
try:
    g["_fp_run"](["sh", "-c", "sleep 307 & sleep 307 & wait"], time.time() + 13, "fake slow build")
    print("FAIL 1: no timeout")
except TimeoutError as e:
    print("1 TimeoutError after %.1fs: %s" % (time.time() - t, e))
time.sleep(0.5)
left = subprocess.run(["pgrep", "-f", "sleep 307"], capture_output=True, text=True).stdout.split()
print("1 leftover sleep processes:", left, "PASS" if not left else "FAIL")

r = g["_fp_run"](["sh", "-c", "echo hi"], time.time() + 30, "fast", capture_output=True, text=True)
print("2 rc", r.returncode, repr(r.stdout), "PASS" if r.returncode == 0 and r.stdout == "hi\n" else "FAIL")

Path("/tmp/v36_src.bin").write_bytes(b"x" * (1 << 20))
try:
    g["_fp_copy"]("/tmp/v36_src.bin", "/tmp/v36_dst.bin", time.time() - 1, "copy")
    print("FAIL 3")
except TimeoutError as e:
    print("3 TimeoutError:", e, "PASS")
