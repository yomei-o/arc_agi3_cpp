"""v42a from v40 (LB 0.345): the pool popularity re-order (0.15 x [ln(1+SID) + ln(1+PMID)] on the pool candidates of
the top 60) applied to every query instead of only when the best library similarity < 0.9.

Public-lineage version table: "pool popularity re-rank z(ranker) + 0.15 x documentation, ungated: +0.010 on the
leaderboard". Ours has been gated (libSim < 0.9) since v27; our holdouts read popularity backwards (hn_STATUS.md), so
this is an LB test.    python make_v42_nb.py [tag] [ppgate]
"""
import ast, json, os, sys
NBD = r"C:\prog\claude4\arc\arc3\casmi\notebook"
SP = r"C:\Users\yomei2\AppData\Local\Temp\claude\C--prog-claude4\7b6059ac-6d96-452f-ab4e-313fe77f5b89\scratchpad"
TAG = sys.argv[1] if len(sys.argv) > 1 else "v42a"
GATE = sys.argv[2] if len(sys.argv) > 2 else "99"
base = json.load(open(os.path.join(NBD, "casmi_match_v40.ipynb"), encoding="utf-8"))
meta = json.load(open(os.path.join(NBD, "kernel-metadata-v40.json"), encoding="utf-8"))
OLD = 'V27_OPT += \\",ppop=0.15,ppgate=0.9,ppopf=\\"'
nb = json.loads(json.dumps(base))
done = 0
for c in nb["cells"]:
    s = "".join(c["source"])
    old = '",ppop=0.15,ppgate=0.9,ppopf="'
    if old in s:
        s = s.replace(old, f'",ppop=0.15,ppgate={GATE},ppopf="')
        done += 1
    c["source"] = s.splitlines(keepends=True)
assert done == 1, done
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
print(TAG, "ppgate", GATE, m["id"])
