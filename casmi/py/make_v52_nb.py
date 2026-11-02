"""v52 = v47 (LB 0.361, rr group form, ice 1.0 / gl 1.0) with the group base = z(ranker score + 0.15 pop) from the
ranker's top-60 dump (ppdet) instead of z(1/(rank+1)) - the 0.468 notebook z-scores a score, not a rank. Groups whose
members lack a dumped score keep the reciprocal-rank base. Weight probes v50a/v50b/v51a (0.360/0.355/0.353) put the
rank-base form at its optimum, so the base is the next free axis.
    python make_v52_nb.py [tag] [base_mult]
"""
import ast, json, os, sys
NBD = r"C:\prog\claude4\arc\arc3\casmi\notebook"
SP = r"C:\Users\yomei2\AppData\Local\Temp\claude\C--prog-claude4\7b6059ac-6d96-452f-ab4e-313fe77f5b89\scratchpad"
TAG = sys.argv[1] if len(sys.argv) > 1 else "v52"
BMUL = sys.argv[2] if len(sys.argv) > 2 else "1.0"
base = json.load(open(os.path.join(NBD, "casmi_match_v47.ipynb"), encoding="utf-8"))
meta = json.load(open(os.path.join(NBD, "kernel-metadata-v47.json"), encoding="utf-8"))

REPL = [
    ('print("v28 options:", V27_OPT, flush=True)\n',
     'V27_OPT += ",ppdet=pp_dump.txt"  # v52: ranker scores for the group base\n'
     'print("v28 options:", V27_OPT, flush=True)\n'),
    ("        fused = _z([1.0 / (i + 1) for i in idx])\n",
     "        _b = [_BASE[i] for i in idx]\n"
     "        if all(x is not None for x in _b):\n"
     "            fused = [BMUL * x for x in _z(_b)]\n"
     "            _NB[0] += 1\n"
     "        else:\n"
     "            fused = _z([1.0 / (i + 1) for i in idx])\n"
     "            _NB[1] += 1\n"),
    ("    moved = top1 = 0\n",
     "    BMUL, _NB = __BMUL__, [0, 0]\n"
     "    _pps = {}\n"
     "    try:\n"
     "        _pp = {}\n"
     "        for l in open(WORK / \"pp_dump.txt\", encoding=\"utf-8\"):\n"
     "            p = l.rstrip(\"\\n\").split(\"\\t\")\n"
     "            if len(p) >= 6 and p[3] == \"1\":\n"
     "                _pp.setdefault(p[5], []).append((p[0], float(p[2]) + 0.15 * float(p[4])))\n"
     "        with open(POOL730_DIR / \"pool730_mols.txt\", encoding=\"utf-8\") as f:\n"
     "            for l in f:\n"
     "                t = l.find(\"\\t\")\n"
     "                if t > 0 and l[:t] in _pp:\n"
     "                    s = l[t + 1:].rstrip(\"\\r\\n\")\n"
     "                    for q, v in _pp[l[:t]]:\n"
     "                        _pps[(q, s)] = v\n"
     "    except Exception as e:\n"
     "        print(\"pp_dump unusable:\", repr(e), flush=True)\n"
     "    print(\"v52 base scores: %d (query, smiles)\" % len(_pps), flush=True)\n"
     "    moved = top1 = 0\n"),
    ("        o = _rerank(smis, forms, [ic, gl], [LAM_ICE, LAM_GL])\n",
     "        _BASE = [_pps.get((mid, s)) for s in smis]\n"
     "        o = _rerank(smis, forms, [ic, gl], [LAM_ICE, LAM_GL])\n"),
    ('    print("FW stage done in',
     '    print("v52 groups: score base %d, rank base %d" % tuple(_NB), flush=True)\n'
     '    print("FW stage done in'),
]
nb = json.loads(json.dumps(base))
cnt = [0] * len(REPL)
for c in nb["cells"]:
    s = "".join(c["source"])
    for k, (a, b) in enumerate(REPL):
        if a in s:
            cnt[k] += s.count(a)
            s = s.replace(a, b.replace("__BMUL__", BMUL))
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
print(TAG, "bmul", BMUL, m["id"])
