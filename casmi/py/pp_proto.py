"""Post-processing prototypes on phys_v27 feature dumps (pp_STATUS / postproc_STATUS.md).

1. official-identity dedup + refill: identity = stereo stripped -> RDKit canonical tautomer -> InChIKey[:14]
2. formula-group rerank: in the final list, candidates sharing a molecular formula (2+ in the list) are
   reordered among the positions the group occupies, by a forward-model score (phys / MLP / TFM / mix),
   or by z(fused) + lam * z(forward) (z over the list), or randomly (control).

The fused score is rebuilt from the dump exactly as phys_v27's hand fusion:
  z(phys) + 0.25 z(mlp) + 1.0 z(rev) + 8 lib^32 + 24 ana + gw . gen   (z over the query's candidates)
and checked against the run's CSV order.

usage: pp_proto.py <dir> <tag> <qry> [tag qry ...]
   reads <dir>/pp/f_<tag>.txt, <dir>/pp/<tag>.csv, <dir>/<qry>_mols.txt; key cache <dir>/pp/keycache.pkl
"""
import csv
import os
import pickle
import random
import sys

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

RDLogger.DisableLog("rdApp.*")
_te = rdMolStandardize.TautomerEnumerator()
GW = [-2, 0, -1, 4, 0, 1.5]
TW = float(os.environ.get("PP_TW", "0"))   # + TW * z(TFM sum-logit) in the fusion (v27b uses 2)
TOPN = 60          # keys/formulas only for the top of each list (refill never reaches further)

DIR = sys.argv[1] if len(sys.argv) > 1 else "."
CACHE = os.path.join(DIR, "pp", "keycache.pkl")
cache = {}
n_new = 0


def mkey(smi):
    """(metric key, formula, plain ik14)"""
    global n_new
    v = cache.get(smi)
    if v is not None:
        return v
    n_new += 1
    v = compute_key(smi)
    cache[smi] = v
    return v


def prefetch(smis, procs=8):
    """compute the uncached keys in parallel (TautomerEnumerator ~0.1 s/molecule)"""
    from multiprocessing import Pool
    todo = sorted(set(s for s in smis if s not in cache))
    if not todo:
        return
    with Pool(procs) as p:
        for s, v in zip(todo, p.map(compute_key, todo, chunksize=16)):
            cache[s] = v
    print(f"  prefetched {len(todo)} keys", flush=True)


def compute_key(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        v = (None, None, None)
    else:
        try:
            ik = Chem.MolToInchiKey(m)[:14] or None
        except Exception:
            ik = None
        try:
            f = CalcMolFormula(m)
        except Exception:
            f = None
        try:
            m2 = Chem.Mol(m)
            Chem.RemoveStereochemistry(m2)
            k = Chem.MolToInchiKey(_te.Canonicalize(m2))[:14] or None
        except Exception:
            k = None
        v = (k, f, ik)
    return v


def z(v):
    v = np.asarray(v, dtype=np.float64)
    s = v.std() + 1e-6
    return (v - v.mean()) / s


def read_dump(path):
    Q = []
    with open(path, encoding="utf-8") as f:
        while True:
            h = f.readline()
            if not h:
                break
            _, name, truth, n = h.split()
            rows = [f.readline().split() for _ in range(int(n))]
            if not rows:
                Q.append(None)
                continue
            a = np.array([[float(x) for x in r[:6]] for r in rows]).reshape(-1, 6)
            g = np.array([[float(x) for x in r[7:13]] for r in rows]).reshape(-1, 6)
            t = np.array([[float(x) for x in r[13:-1]] for r in rows]).reshape(len(rows), -1)
            Q.append(dict(name=name, truth=int(truth), phys=a[:, 0], mlp=a[:, 1], rev=a[:, 2], lib=a[:, 3],
                          ana=a[:, 4], gen=g, tfm=t, cname=[r[6] for r in rows], smi=[r[-1] for r in rows]))
    return Q


def fused(q):
    return (z(q["phys"]) + 0.25 * z(q["mlp"]) + 1.0 * z(q["rev"]) + 8 * q["lib"] ** 32 + 24 * q["ana"]
            + q["gen"] @ np.array(GW) + (TW * z(q["tfm"][:, 0]) if TW and q["tfm"].shape[1] else 0.0))


def mrr(ranks):
    return float(np.mean([1.0 / r if 0 < r <= 25 else 0.0 for r in ranks]))


def boot(a, b, n=2000, seed=1):
    a, b = np.asarray(a), np.asarray(b)
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(a), len(a))
        d.append(a[i].mean() - b[i].mean())
    d = np.sort(d)
    return d[int(0.025 * n)], d[int(0.975 * n)]


def rr_list(lst, hit):
    for i, c in enumerate(lst[:25]):
        if hit(c):
            return 1.0 / (i + 1)
    return 0.0


def group_rerank(lst, key_of, form_of, fwd, mode, lam, fz, rng=None):
    """lst: candidate indices in order; reorder within formula groups (2+ members) keeping positions."""
    groups = {}
    for pos, c in enumerate(lst):
        f = form_of(c)
        if f is None:
            continue
        groups.setdefault(f, []).append(pos)
    out = list(lst)
    for f, pos in groups.items():
        if len(pos) < 2:
            continue
        mem = [lst[p] for p in pos]
        if mode == "rand":
            rng.shuffle(mem)
        else:
            mem.sort(key=lambda c: -(fwd[c] if lam is None else fz[c] + lam * fwd[c]))
        for p, c in zip(pos, mem):
            out[p] = c
    return out


def run(tag, qry):
    Q = read_dump(os.path.join(DIR, "pp", f"f_{tag}.txt"))
    truth_smi = {}
    with open(os.path.join(DIR, f"{qry}_mols.txt"), encoding="utf-8") as f:
        for l in f:
            p = l.rstrip("\r\n").split("\t")
            if len(p) > 1:
                truth_smi[p[0]] = p[1]
    # check the rebuilt fusion against the run's CSV (names)
    csvrows = {}
    with open(os.path.join(DIR, "pp", f"{tag}.csv"), encoding="utf-8") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            csvrows[row[0]] = row[1].split(";")
    agree = tot = 0
    res = {}
    prefetch([q["smi"][c] for q in Q if q is not None for c in np.argsort(-fused(q), kind="stable")[:TOPN]]
             + list(truth_smi.values()))
    has_tfm = next(q for q in Q if q is not None)["tfm"].shape[1] > 0
    rng = random.Random(7)
    nq = len(Q)
    Q = [q for q in Q if q is not None]
    for q in Q:
        f = fused(q)
        order = list(np.argsort(-f, kind="stable"))
        names = [q["cname"][i] if not q["cname"][i].startswith("GEN") else None for i in order[:25]]
        cs = csvrows.get(q["name"], [])
        for a, b in zip(names, cs):
            if a is not None:
                tot += 1
                agree += a == b
        top = order[:TOPN]
        K = {c: mkey(q["smi"][c]) for c in top}
        tk = mkey(truth_smi[q["name"]])[0] if q["name"] in truth_smi else None
        t = q["truth"]
        hit_name = lambda c: c == t
        hit_metric = lambda c: (tk is not None and K[c][0] == tk) or c == t
        res.setdefault("base_name", []).append(rr_list(order, hit_name))
        res.setdefault("base_metric", []).append(rr_list(top, hit_metric))
        seen, dd = set(), []
        for c in top:
            k = K[c][0] or ("raw:" + q["smi"][c])
            if k in seen:
                continue
            seen.add(k)
            dd.append(c)
        res.setdefault("dedup_metric", []).append(rr_list(dd, hit_metric))
        ks = [K[c][0] for c in top[:25]]
        res.setdefault("_wasted", []).append(len(ks) - len(set(ks)))
        iks = [K[c][2] for c in top[:25]]
        res.setdefault("_wasted_ik", []).append(len(iks) - len(set(iks)))
        res.setdefault("dedup_name", []).append(rr_list(dd, hit_name))
        # formula-group rerank on the deduped list
        fz_all = f
        fwds = {"phys": z(q["phys"]), "mlp": z(q["mlp"]), "rev": z(q["rev"])}
        fwds["ps"] = fwds["phys"] + fwds["mlp"]
        if has_tfm:
            fwds["tfm"] = z(q["tfm"][:, 0])
            fwds["tfmT"] = z(q["tfm"][:, 1])
            fwds["mix"] = fwds["phys"] + fwds["mlp"] + fwds["tfm"]
        for N in (25, 40):
            lst = dd[:N]
            fl = np.array([fz_all[c] for c in lst])
            fzl = dict(zip(lst, z(fl)))
            form_of = lambda c: K[c][1]
            for fn, fw in fwds.items():
                wl = np.array([fw[c] for c in lst])
                fwl = dict(zip(lst, z(wl)))
                for lam in (None, 0.25, 0.5, 1.0):
                    o = group_rerank(lst, None, form_of, fwl, "sort", lam, fzl)
                    res.setdefault(f"fgr N{N} {fn} {'pure' if lam is None else 'lam' + str(lam)}", []).append(
                        rr_list(o, hit_metric))
            for s in range(5):
                o = group_rerank(lst, None, form_of, None, "rand", None, fzl, rng)
                res.setdefault(f"fgr N{N} rand s{s}", []).append(rr_list(o, hit_metric))
        # group statistics
        lst = dd[:25]
        forms = [K[c][1] for c in lst]
        ng = sum(1 for fm in set(forms) if forms.count(fm) >= 2)
        res.setdefault("_groups", []).append(ng)
        res.setdefault("_ingroup", []).append(sum(1 for fm in forms if forms.count(fm) >= 2))
        tpos = next((i for i, c in enumerate(lst) if hit_metric(c)), None)
        res.setdefault("_truth_in_group", []).append(
            1 if tpos is not None and forms.count(forms[tpos]) >= 2 else 0)
        res.setdefault("_truth_in_list", []).append(1 if tpos is not None else 0)
    for k in res:
        res[k] += [0] * (nq - len(Q))
    print(f"== {tag} ({qry}) n={nq} (empty {nq - len(Q)})  fusion rebuilt vs CSV: {agree}/{tot} names agree")
    base = res["dedup_metric"]
    print(f"  groups(2+) per list {np.mean(res['_groups']):.2f}, candidates in groups {np.mean(res['_ingroup']):.1f}/25, "
          f"truth in top25 {sum(res['_truth_in_list'])}, of which in a group {sum(res['_truth_in_group'])}")
    w = np.array(res["_wasted"])
    print(f"  duplicates in top25 (metric identity): lists with any {(w > 0).sum()}, wasted slots {w.sum()} "
          f"({w.sum() / (25 * nq):.2%}); plain-ik14 duplicates {np.sum(res['_wasted_ik'])}")
    for k in ("base_name", "base_metric", "dedup_name", "dedup_metric"):
        print(f"  {k:14s} {np.mean(res[k]):.4f}")
    lo, hi = boot(res["dedup_metric"], res["base_metric"])
    print(f"  dedup - base (metric) {np.mean(res['dedup_metric']) - np.mean(res['base_metric']):+.4f} [{lo:+.4f}, {hi:+.4f}]")
    for N in (25, 40):
        rnd = np.mean([np.mean(res[f"fgr N{N} rand s{s}"]) for s in range(5)])
        print(f"  fgr N{N} control random (5 seeds) {rnd:.4f} ({rnd - np.mean(base):+.4f})")
        for k in sorted(res):
            if k.startswith(f"fgr N{N} ") and "rand" not in k:
                v = np.mean(res[k])
                lo, hi = boot(res[k], base)
                print(f"  {k:28s} {v:.4f} ({v - np.mean(base):+.4f} [{lo:+.4f}, {hi:+.4f}])")
    return res


if __name__ == "__main__":
    if os.path.exists(CACHE):
        cache = pickle.load(open(CACHE, "rb"))
    out = {}
    a = sys.argv[2:]
    for i in range(0, len(a), 2):
        out[a[i]] = run(a[i], a[i + 1])
        sys.stdout.flush()
    pickle.dump(cache, open(CACHE, "wb"))
    pickle.dump(out, open(os.path.join(DIR, "pp", "proto_res_" + "_".join(a[0::2]) + ".pkl"), "wb"))
    print(f"new keys computed {n_new}")
