"""Build notebook/casmi_match_v28.ipynb from notebook/casmi_match.ipynb (v27a):
  * the ranker is phys_v28fgr.cpp (phys_v27 + formula-group rerank option + nout)
  * the ranker writes 40 candidates per query (nout=40) so a dedup can refill
  * the final list is deduplicated by the competition's identity (stereo stripped -> RDKit canonical
    tautomer -> InChIKey first block) when an RDKit with MolStandardize can be imported (or installed
    from an attached wheel); otherwise the old exact-string dedup is kept and the log says DEDUP DISABLED.
usage: pp_make_nb.py <repo casmi dir> [FGR option string, e.g. ",fgr=tfm,fgrn=40,fgrl=0.5"]
"""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
fgr = sys.argv[2] if len(sys.argv) > 2 else ""
src = root / "notebook" / "casmi_match.ipynb"
dst = root / "notebook" / "casmi_match_v28.ipynb"
nb = json.loads(src.read_text(encoding="utf-8"))


def cell_text(i):
    return "".join(nb["cells"][i]["source"])


def set_text(i, s):
    nb["cells"][i]["source"] = s.splitlines(keepends=True)


def rep(s, a, b):
    assert a in s, a[:80]
    return s.replace(a, b, 1)


c6 = cell_text(6)
c6 = rep(c6, 'srcs_fp = ["phys_v27.cpp"] + srcs[1:]',
         'srcs_fp = ["phys_v28fgr.cpp"] + srcs[1:]  # v28: phys_v28fgr = phys_v27 + formula-group rerank option + nout')
set_text(6, c6)

c7 = cell_text(7)
c7 = rep(c7, 'V27_OPT = ",libcap=1" if FP_ARGS else ""',
         'V27_OPT = ",libcap=1" if FP_ARGS else ""\n'
         '# v28: 40 candidates per query (a queue for the identity dedup below) and the formula-group rerank option\n'
         f'V28_OPT = (",nout=40{fgr}") if FP_ARGS else ""\n'
         'print("v28 options:", V28_OPT, flush=True)')
c7 = rep(c7, '"gen=1,cap=300,prior=gen2_prior_c.txt,gw=-2:0:-1:4:0:1.5,thr=4,grid=0" + V27_OPT]',
         '"gen=1,cap=300,prior=gen2_prior_c.txt,gw=-2:0:-1:4:0:1.5,thr=4,grid=0" + V27_OPT + V28_OPT]')

DEDUP = r'''
# v28: deduplicate the final list by the competition's identity - stereochemistry removed, then RDKit's
# canonical tautomer, then the InChIKey first block (14 characters) - and refill the freed slots from the
# rest of the queue (ranker candidates 26-40, then the library list). Two SMILES that are stereo variants
# or tautomers of one structure score as the same answer, so the second one only wastes a slot.
# Holdouts (postproc_STATUS.md): 0.3-0.8% of slots, MRR +0.001.
def _rdkit_ok():
    try:
        from rdkit import Chem  # noqa: F401
        from rdkit.Chem.MolStandardize import rdMolStandardize  # noqa: F401
        return True
    except Exception as e:
        print("rdkit import failed:", repr(e), flush=True)
    tag = "cp%d%d" % sys.version_info[:2]
    whl = []
    for pat in ("*/rdkit-*-%s-*.whl" % tag, "*/*/rdkit-*-%s-*.whl" % tag, "*/*/*/rdkit-*-%s-*.whl" % tag):
        whl += sorted(Path("/kaggle/input").glob(pat))
    print("rdkit wheels:", [str(w) for w in whl], flush=True)
    if not whl:
        return False
    rr = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", str(whl[0])],
                        capture_output=True, text=True)
    print("rdkit pip rc", rr.returncode, rr.stderr[-500:], flush=True)
    try:
        import importlib
        importlib.invalidate_caches()
        from rdkit import Chem  # noqa: F401
        from rdkit.Chem.MolStandardize import rdMolStandardize  # noqa: F401
        return True
    except Exception as e:
        print("rdkit import after install failed:", repr(e), flush=True)
        return False


_TE = None
def _ident(smi):
    global _TE
    from rdkit import Chem, RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize
    RDLogger.DisableLog("rdApp.*")
    if _TE is None:
        _TE = rdMolStandardize.TautomerEnumerator()
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        Chem.RemoveStereochemistry(m)
        return Chem.MolToInchiKey(_TE.Canonicalize(m))[:14] or None
    except Exception:
        return None


queues = {}
for mol_id, dsmi in direct.items():
    d_list = [s for s in dsmi.split(";") if s]
    p_list = [s for s in pool_c.get(mol_id, "").split(";") if s]
    q, seen = [], set()
    for s in p_list + d_list:
        if s not in seen:
            seen.add(s); q.append(s)
    queues[mol_id] = q
t5 = time.time()
DEDUP_OK = False
try:
    DEDUP_OK = _rdkit_ok()
except Exception as e:
    print("rdkit check failed:", repr(e), flush=True)
dedup_rows = {}
if DEDUP_OK:
    try:
        from multiprocessing import Pool as _Pool
        from rdkit import rdBase
        HEAD = 32   # keys for the head of each queue; a row needing more is extended one by one
        need = sorted({s for q in queues.values() for s in q[:HEAD]})
        with _Pool(4) as pp:
            ident = dict(zip(need, pp.map(_ident, need, chunksize=32)))
        n_changed = n_refilled = n_extra = 0
        for mol_id, q in queues.items():
            out, seen = [], set()
            for i, s in enumerate(q):
                if s not in ident:
                    ident[s] = _ident(s); n_extra += 1
                k = ident[s] or ("raw:" + s)
                if k in seen:
                    continue
                seen.add(k); out.append(s)
                if len(out) == K:
                    break
            if out[:K] != q[:K]:
                n_changed += 1
                n_refilled += sum(1 for s in out if s not in set(q[:K]))
            dedup_rows[mol_id] = out
        print("dedup (rdkit %s): %d keys in %.0fs (+%d extra), rows changed %d / %d, slots refilled %d"
              % (rdBase.rdkitVersion, len(need), time.time() - t5, n_extra, n_changed, len(queues), n_refilled),
              flush=True)
    except Exception as e:
        print("DEDUP DISABLED (error):", repr(e), flush=True)
        dedup_rows = {}
else:
    print("DEDUP DISABLED (no rdkit with MolStandardize)", flush=True)
'''

old_loop = '''    order = p_list + d_list
    merged, seen = [], set()
    for s in order:
        if s not in seen:
            seen.add(s); merged.append(s)
        if len(merged) == K: break'''
new_loop = '''    order = p_list + d_list
    merged, seen = [], set()
    for s in order:
        if s not in seen:
            seen.add(s); merged.append(s)
        if len(merged) == K: break
    if mol_id in dedup_rows:  # v28: identity dedup + refill
        merged = dedup_rows[mol_id][:K]'''
c7 = rep(c7, old_loop, new_loop)
c7 = rep(c7, "K, THETA = 25, 0.9\n", "K, THETA = 25, 0.9\n" + DEDUP)
c7 = rep(c7, '# v27: physics params', '# v28: + identity dedup of the final list, formula-group rerank option (postproc_STATUS.md)\n# v27: physics params')
set_text(7, c7)
dst.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", dst)
