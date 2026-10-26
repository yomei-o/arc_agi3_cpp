"""auto_sub_remote.py: build-machine twin of scratchpad pe/auto_sub4.sh (survives the laptop shutting down).

    python auto_sub_remote.py <kernel slug> <version> <out dir name> <message> <required log pattern>...
Waits for the kernel run (<= 9 h), downloads its log + submission.csv to C:\\prog\\casmi\\<out>, checks the required
log lines ("fpnet stage done", "phys_rank ran in", "final: rows" + the given ones, no "DISABLED"), then submits the
kernel output through competition_submit_cli. Prints CHECKS OK - submitted / CHECKS FAILED / SUBMISSION REJECTED.
"""
import re, shutil, subprocess, sys, time
from pathlib import Path

slug, ver, out, msg = sys.argv[1:5]
extra = sys.argv[5:]
COMP = "enveda-CASMI26-molecule-id-mass-spectra"
OUT = Path(r"C:\prog\casmi") / out


def log(s):
    print(time.strftime("%H:%M"), s, flush=True)


done_ok = False
for i in range(360):
    s = subprocess.run(["kaggle", "kernels", "status", "yomeiotani/" + slug], capture_output=True, text=True)
    st = (s.stdout + s.stderr).strip().splitlines()[-1] if (s.stdout + s.stderr).strip() else ""
    if "COMPLETE" in st:
        done_ok = True
        break
    if "ERROR" in st or "CANCEL" in st:
        log("KERNEL FAILED: " + st)
        sys.exit(1)
    time.sleep(90)
if not done_ok:
    log("TIMED OUT waiting for %s - not submitting" % slug)
    sys.exit(1)
log(slug + " complete")
shutil.rmtree(OUT, ignore_errors=True)
OUT.mkdir(parents=True)
r = subprocess.run(["kaggle", "kernels", "output", "yomeiotani/" + slug, "-p", str(OUT),
                    "--file-pattern", r"(\.log|submission\.csv)$", "-o"], capture_output=True, text=True)
print((r.stdout + r.stderr)[-400:], flush=True)
logs = list(OUT.glob("*.log"))
text = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in logs)
for pat in (r"v27 params:[^\"]*", r"fpnet stage done[^\"]*", r"phys_rank ran in[^\"]*", r"OWN FPNET stage done[^\"]*"):
    for m in re.findall(pat, text)[:2]:
        print("  " + m[:160], flush=True)
ok = True
for p in ["fpnet stage done", "phys_rank ran in", "final: rows"] + extra:
    if p not in text:
        log("MISSING: " + p)
        ok = False
if "DISABLED" in text:
    log("DISABLED found")
    ok = False
if not (OUT / "submission.csv").exists():
    log("submission.csv missing")
    ok = False
if not ok:
    log("CHECKS FAILED - not submitting")
    sys.exit(1)
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
try:
    res = api.competition_submit_cli(file_name="submission.csv", message=msg, competition=COMP,
                                     kernel="yomeiotani/" + slug, version=ver)
    log("RESULT " + str(res))
    log("CHECKS OK - submitted %s v%s" % (slug, ver))
except Exception as e:
    log("SUBMISSION REJECTED: " + repr(e)[:600])
    body = getattr(getattr(e, "response", None), "text", None)
    if body:
        print(str(body)[:600], flush=True)
    sys.exit(1)
