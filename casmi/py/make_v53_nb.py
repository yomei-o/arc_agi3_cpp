"""v53 variants of v47 (LB 0.361, rr group form, ice 1.0 / gl 1.0, T4):
  v53  : group base z(1/(i+K)) with K=3 (the 0.468 notebook's RRF shape 1/(3+r)) instead of z(1/(i+1)): a flatter top
         gives the forward models more say over ranks 1-3.
  v53b : v47 + gated E2R2 - library top-1 inserted at rank 2 when match.cpp's best score >= 1.0 (rf_STATUS: holdout
         mixture +0.008, xinstr 0; on the LB only ever tested together with LambdaMART in v31).
    python make_v53_nb.py v53 base3 | python make_v53_nb.py v53b e2r2
"""
import ast, json, os, sys
NBD = r"C:\prog\claude4\arc\arc3\casmi\notebook"
SP = r"C:\Users\yomei2\AppData\Local\Temp\claude\C--prog-claude4\7b6059ac-6d96-452f-ab4e-313fe77f5b89\scratchpad"
TAG, MODE = sys.argv[1], sys.argv[2]
base = json.load(open(os.path.join(NBD, "casmi_match_v47.ipynb"), encoding="utf-8"))
meta = json.load(open(os.path.join(NBD, "kernel-metadata-v47.json"), encoding="utf-8"))

if MODE == "base3":
    REPL = [("        fused = _z([1.0 / (i + 1) for i in idx])\n",
             "        fused = _z([1.0 / (i + 3) for i in idx])  # v53: RRF shape 1/(3+r)\n")]
elif MODE == "e2r2":
    REPL = [("    order = p_list + d_list\n",
             "    order = p_list + d_list\n"
             "    if d_list and best.get(mol_id, 0.0) >= 1.0 and d_list[0] not in p_list[:2]:  # v53b: gated E2R2\n"
             "        order = p_list[:1] + [d_list[0]] + p_list[1:] + d_list\n"
             "        n_e2r2 = globals().get('n_e2r2', 0) + 1\n"),
            ("merged_dict = dict(merged_rows)\n",
             "merged_dict = dict(merged_rows)\n"
             "print('E2R2 done: library top-1 placed at rank 2 for %d queries' % globals().get('n_e2r2', 0), flush=True)\n")]
else:
    raise SystemExit("mode?")
nb = json.loads(json.dumps(base))
cnt = [0] * len(REPL)
for c in nb["cells"]:
    s = "".join(c["source"])
    for k, (a, b) in enumerate(REPL):
        if a in s:
            cnt[k] += s.count(a)
            s = s.replace(a, b)
    c["source"] = s.splitlines(keepends=True)
assert cnt == [1] * len(REPL), cnt
for c in nb["cells"]:
    if c["cell_type"] == "code":
        ast.parse("\n".join(l for l in "".join(c["source"]).splitlines() if not l.startswith(("!", "%"))))
name = f"casmi_match_{TAG}.ipynb"
m = dict(meta, id=f"yomeiotani/casmi26-cpp-library-search-{TAG}", title=f"CASMI26 cpp library search {TAG}",
         code_file=name)
for d in (NBD, os.path.join(SP, "kp_" + TAG)):
    os.makedirs(d, exist_ok=True)
    json.dump(nb, open(os.path.join(d, name), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
json.dump(m, open(os.path.join(NBD, f"kernel-metadata-{TAG}.json"), "w", encoding="utf-8"), indent=2)
json.dump(m, open(os.path.join(SP, "kp_" + TAG, "kernel-metadata.json"), "w", encoding="utf-8"), indent=2)
print(TAG, MODE, m["id"])
