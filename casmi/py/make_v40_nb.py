"""v40 from v36nofp (LB 0.344): own FPNet pool term in continuous-Tanimoto form instead of the logit dot product.

models38_STATUS: Tanimoto form (p = sigmoid(logit), CNN/FT form) beats the dot form by +0.005..0.01 summed over the timsTOF
holdouts. The mounted dataset still has the dot-only fpnet_pool.cpp, so the notebook writes cpp/fpnet/fpnet_pool.cpp
(dot|tan) to WORK and compiles it with g++ (a prebuilt miss = plain compile, ~2 s).
    python mk_v40.py [tag] [fpnet weight]
"""
import ast, json, os, sys
NBD = r"C:\prog\claude4\arc\arc3\casmi\notebook"
SRC = r"C:\prog\claude4\arc\arc3\casmi\cpp\fpnet\fpnet_pool.cpp"
SP = r"C:\Users\yomei2\AppData\Local\Temp\claude\C--prog-claude4\7b6059ac-6d96-452f-ab4e-313fe77f5b89\scratchpad"
TAG = sys.argv[1] if len(sys.argv) > 1 else "v40"
FW = sys.argv[2] if len(sys.argv) > 2 else "1.0"
base = json.load(open(os.path.join(NBD, "casmi_match_v36nofp.ipynb"), encoding="utf-8"))
meta = json.load(open(os.path.join(NBD, "kernel-metadata-v36nofp.json"), encoding="utf-8"))
cpp = open(SRC, encoding="utf-8").read()
assert "tan" in cpp and '"""' not in cpp

OLD_RUN = '        r = _fp_run([str(FB / "fpnet_pool"), str(WORK), "test", str(WORK / "fpnet_test_mol.bin"), "pool730",\n' \
          '                     str(FP_USE), "10", str(WORK / "fpnet_test.txt")], FP_DEADLINE, "fpnet_pool",\n'
NEW_RUN = '        # v40: Tanimoto form (fpnet_pool dot|tan, source embedded below; the mounted copy has dot only)\n' \
          '        FPP_SRC = WORK / "fpnet_pool_tan.cpp"\n' \
          '        FPP_SRC.write_text(FPNET_POOL_TAN_SRC)\n' \
          '        r = _fp_run(["g++", "-O2", "-std=c++17", "-o", str(WORK / "fpnet_pool_tan"), str(FPP_SRC)], FP_DEADLINE,\n' \
          '                    "fpnet_pool_tan compile", capture_output=True, text=True)\n' \
          '        if r.returncode != 0:\n' \
          '            print(r.stdout[-2000:], r.stderr[-3000:]); raise RuntimeError("fpnet_pool_tan compile failed")\n' \
          '        r = _fp_run([str(WORK / "fpnet_pool_tan"), str(WORK), "test", str(WORK / "fpnet_test_mol.bin"), "pool730",\n' \
          '                     str(FP_USE), "10", str(WORK / "fpnet_test.txt"), "tan"], FP_DEADLINE, "fpnet_pool tan",\n'
DEF_AT = "def _fp_run(cmd, deadline, what, **kw):\n"
DEF_SRC = "FPNET_POOL_TAN_SRC = " + repr(cpp) + "\n\n\n" + DEF_AT
FPW = ['FP_ARGS = ["fpnet_test.txt", "1.0"]', '"+ 1.0 z(fpnet) (own weights)"']

nb = json.loads(json.dumps(base))
done = {"run": 0, "def": 0, "fpw": 0}
for c in nb["cells"]:
    s = "".join(c["source"])
    if OLD_RUN in s:
        s = s.replace(OLD_RUN, NEW_RUN); done["run"] += 1
        s = s.replace("OWN FPNET stage done in %.0fs: + 1.0 z(fpnet)", "OWN FPNET stage done in %.0fs: + " + FW + " z(fpnet, tanimoto)")
    if DEF_AT in s:
        s = s.replace(DEF_AT, DEF_SRC); done["def"] += 1
    if FW != "1.0":
        for old in FPW:
            if old in s:
                s = s.replace(old, old.replace("1.0", FW)); done["fpw"] += 1
    c["source"] = s.splitlines(keepends=True)
assert done["run"] == 1 and done["def"] == 1, done
for c in nb["cells"]:
    if c["cell_type"] == "code":
        ast.parse("\n".join(l for l in "".join(c["source"]).splitlines() if not l.startswith(("!", "%"))))
name = f"casmi_match_{TAG}.ipynb"
m = dict(meta, id=f"yomeiotani/casmi26-cpp-library-search-{TAG}", title=f"CASMI26 cpp library search {TAG}", code_file=name)
for d in (NBD, os.path.join(SP, "kp_" + TAG)):
    os.makedirs(d, exist_ok=True)
    json.dump(nb, open(os.path.join(d, name), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
json.dump(m, open(os.path.join(NBD, f"kernel-metadata-{TAG}.json"), "w", encoding="utf-8"), indent=2)
json.dump(m, open(os.path.join(SP, "kp_" + TAG, "kernel-metadata.json"), "w", encoding="utf-8"), indent=2)
print(TAG, done, m["id"])
