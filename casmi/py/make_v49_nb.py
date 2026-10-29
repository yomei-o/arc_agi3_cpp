"""v49 = v47 (LB 0.361: own C++ ICE 1.0 / GL 1.0 same-formula re-rank, T4) + PubChem rank-1 promotion as a final step.

The public lineage's biggest single step (+0.024..0.037) promoted the PubChem channel's top pick to rank 1 on strong
evidence. Our ranker has the same rule (pcv=2: relz > relp or (S > sp and pop_top >= popp), gate libSim < 0.9) but
alloc=boltz switches it off. Offline (pc_promo.py) it was worth only +0.001..0.003, but the holdouts have ranked the
forward-model forms backwards, so this is an LB probe. The ranker dumps the per-query stats (pcdet=pc_det.txt: Q line,
V2 line, then the listed PubChem candidates in list order); the top pick is moved to (or inserted at) rank 1 of the
final merged list.
    python make_v49_nb.py [tag] [relp] [sp] [popp]
"""
import ast, json, os, sys
NBD = r"C:\prog\claude4\arc\arc3\casmi\notebook"
SP = r"C:\Users\yomei2\AppData\Local\Temp\claude\C--prog-claude4\7b6059ac-6d96-452f-ab4e-313fe77f5b89\scratchpad"
TAG = sys.argv[1] if len(sys.argv) > 1 else "v49"
RELP = sys.argv[2] if len(sys.argv) > 2 else "3"
SPV = sys.argv[3] if len(sys.argv) > 3 else "6"
POPP = sys.argv[4] if len(sys.argv) > 4 else "5"
base = json.load(open(os.path.join(NBD, "casmi_match_v47.ipynb"), encoding="utf-8"))
meta = json.load(open(os.path.join(NBD, "kernel-metadata-v47.json"), encoding="utf-8"))

OPT_ANCHOR = 'print("v28 options:", V27_OPT, flush=True)\n'
OPT_NEW = 'V27_OPT += ",pcdet=pc_det.txt"  # v49: PubChem gate stats for the rank-1 promotion\n' + OPT_ANCHOR
ANCHOR = "merged_dict = dict(merged_rows)\n"
BLOCK = ANCHOR + '''
# ---- v49: PubChem rank-1 promotion (relz > RELP or (S > SPV and pop_top >= POPP), libSim < 0.9) ---------------------
RELP, SPV, POPP = __RELP__, __SPV__, __POPP__
try:
    _det, _q, _st = {}, None, None
    for _l in open(WORK / "pc_det.txt", encoding="utf-8"):
        _p = _l.rstrip("\\n").split("\\t")
        if _p[0] == "Q":
            _q = _p[1]
            _det[_q] = {"lib": float(_p[2]), "v2": None, "top": None}
        elif _p[0] == "V2" and _q is not None:
            _det[_q]["v2"] = (float(_p[3]), float(_p[4]), float(_p[5]))
        elif _q is not None and len(_p) >= 12 and _det[_q]["top"] is None:
            _det[_q]["top"] = _p[11]
    _np = _nc = 0
    for _m, _d in _det.items():
        if _d["v2"] is None or not _d["top"] or _d["lib"] >= 0.9 or _m not in merged_dict:
            continue
        _relz, _S, _pop = _d["v2"]
        if not (_relz > RELP or (_S > SPV and _pop >= POPP)):
            continue
        _L = [s for s in merged_dict[_m].split(";") if s]
        _t = _d["top"]
        _nc += bool(_L) and _L[0] != _t
        _L = [_t] + [s for s in _L if s != _t]
        merged_dict[_m] = ";".join(_L[:25])
        _np += 1
    print("PC PROMO done: %d queries in pc_det, promoted %d (top-1 changed %d), relp %g sp %g popp %g"
          % (len(_det), _np, _nc, RELP, SPV, POPP), flush=True)
except Exception as e:
    print("PC PROMO DISABLED:", repr(e), flush=True)
# ---- end v49 block -----------------------------------------------------------------------------------------------
'''
BLOCK = BLOCK.replace("__RELP__", RELP).replace("__SPV__", SPV).replace("__POPP__", POPP)
nb = json.loads(json.dumps(base))
done = [0, 0]
for c in nb["cells"]:
    s = "".join(c["source"])
    if OPT_ANCHOR in s:
        s = s.replace(OPT_ANCHOR, OPT_NEW); done[0] += 1
    if ANCHOR in s:
        s = s.replace(ANCHOR, BLOCK); done[1] += 1
    c["source"] = s.splitlines(keepends=True)
assert done == [1, 1], done
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
print(TAG, "relp", RELP, "sp", SPV, "popp", POPP, m["id"])
