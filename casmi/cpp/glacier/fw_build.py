"""fw_build.py: compile fw_score (own C++ GLACIER + ICEBERG re-scorer) with g++, object-wise in parallel.

    python fw_build.py <fwm src dir> <rdkit-min dir> <obj dir> <out exe> [jobs=4]
Same function as embedded in the notebook (build_fw_score). Returns 0 on success.
"""
import os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def build_fw_score(src, rdk, obj, out, jobs=4, extra=()):
    src, rdk, obj = Path(src), Path(rdk), Path(obj)
    obj.mkdir(parents=True, exist_ok=True)
    files = [src / "fw_score.cpp", src / "GraphMol" / "PartialCharges" / "GasteigerCharges.cpp",
             src / "GraphMol" / "PartialCharges" / "GasteigerParams.cpp"]
    for sub in ("RDGeneral", "DataStructs", "Geometry", "GraphMol", "GraphMol/CIPLabeler",
                "GraphMol/CIPLabeler/configs", "GraphMol/CIPLabeler/rules", "GraphMol/SmilesParse"):
        files += sorted((rdk / sub).glob("*.cpp"))
    flags = ["-O3", "-std=c++20", "-pthread", "-w", "-mavx2", "-mfma", "-DYY_NO_UNISTD_H",
             "-I", str(src), "-I", str(rdk), "-I", str(rdk / "boost-min")] + list(extra)

    def comp(f):
        o = obj / (str(f.relative_to(f.anchor)).replace(os.sep, "_").replace("/", "_").replace(":", "") + ".o")
        if o.exists() and o.stat().st_mtime >= f.stat().st_mtime:
            return o, 0, ""
        r = subprocess.run(["g++"] + flags + ["-c", str(f), "-o", str(o)], capture_output=True, text=True)
        return o, r.returncode, r.stderr[-1500:]

    t = time.time()
    with ThreadPoolExecutor(jobs) as ex:
        res = list(ex.map(comp, files))
    bad = [(o, e) for o, rc, e in res if rc != 0]
    if bad:
        print("fw_score compile errors:", bad[0][0], bad[0][1], flush=True)
        return 1
    r = subprocess.run(["g++", "-pthread", "-o", str(out)] + [str(o) for o, _, _ in res], capture_output=True, text=True)
    print("fw_score build rc %d: %d files in %.0fs" % (r.returncode, len(files), time.time() - t), r.stderr[-1500:],
          flush=True)
    return r.returncode


if __name__ == "__main__":
    sys.exit(build_fw_score(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4],
                            int(sys.argv[5]) if len(sys.argv) > 5 else 4))
