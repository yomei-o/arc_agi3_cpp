"""Assemble and submit a Kaggle zip for the quantum circuit compilation competition (build machine).
    python qcc_submit.py <invsqrt.kmb> <dqi.kmb> <dfthcsos_dir> "<message>" [--dry]
dfthcsos_dir must hold angle_k_*.kmb and submission.npz. The zip is written as submission.zip (the API upload that
worked; the CLI submit returned 400)."""
import os, shutil, sys, zipfile

inv, dqi, chem, msg = sys.argv[1:5]
root = r"C:\prog\qcc\subzip"
if os.path.isdir(root):
    shutil.rmtree(root)
os.makedirs(os.path.join(root, "dfthcsos"))
shutil.copy(inv, os.path.join(root, "invsqrt.kmb"))
shutil.copy(dqi, os.path.join(root, "dqi.kmb"))
names = sorted(f for f in os.listdir(chem) if f.startswith("angle_k_") or f == "submission.npz")
for f in names:
    shutil.copy(os.path.join(chem, f), os.path.join(root, "dfthcsos", f))
out = r"C:\prog\qcc\submission.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for r, _, fs in os.walk(root):
        for f in fs:
            p = os.path.join(r, f)
            z.write(p, os.path.relpath(p, root))
print("zip", os.path.getsize(out), "bytes:", zipfile.ZipFile(out).namelist())
if "--dry" not in sys.argv:
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    print(api.competition_submit(out, msg, "google-verifiable-quantum-circuit-compilation"))
