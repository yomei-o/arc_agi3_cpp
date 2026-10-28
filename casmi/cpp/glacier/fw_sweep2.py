"""fw_sweep2.py: offline sweep of the forward-model terms INSIDE the fused score (the v44 form, now with ICEBERG too),
from the e2e v44 work dir: pp_dump.txt (fused top-60 scores), fw_out_gi.tsv (GPU fw_score: gl and ice per candidate).

    python fw_sweep2.py <work_v44 dir> <truth.tsv> <e2e_score.py dir>
s' = z(score + 0.15 pop) + w_gl z(gl) + w_ice z(ice) over the dumped candidates (missing -> 0), the pool candidates of
the final list re-ordered among their own slots (as the v44 notebook block), merged with direct_submission as fw_sweep.py.
"""
import csv, importlib.util, math, os, sys



def rd(p):
    return {r[0]: r[1] for r in csv.reader(open(p, encoding="utf-8")) if r and r[0] != "molecule_id"}


def z(vals):
    ok = [v is not None and math.isfinite(v) for v in vals]
    xs = [v for v, o in zip(vals, ok) if o]
    if len(xs) < 2:
        return [0.0] * len(vals)
    mu = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / len(xs))
    if not sd > 0:
        return [0.0] * len(vals)
    return [((v - mu) / sd) if o else 0.0 for v, o in zip(vals, ok)]


def mrr(lists, mids=None):
    out = {s: [] for s in sets}
    for m, t in truth.items():
        L = lists.get(m, [])[:25]
        rr = 0.0
        for i, s in enumerate(L):
            k = K[s][0] if s in K else None
            if k is not None and k == tkey[m]:
                rr = 1.0 / (i + 1)
                break
        out[t[1]].append(rr)
    return {s: sum(v) / max(1, len(v)) for s, v in out.items()}, out


def run(wg, wi):
    lists = {}
    for mid, d in direct.items():
        L = [s for s in pool.get(mid, "").split(";") if s]
        v = pp.get(mid)
        if v and (wg or wi):
            smi = [name2smi.get(e[4]) if e[2] == 1 else None for e in v]
            g = [gl.get((mid, s)) if s else None for s in smi]
            ic = [ice.get((mid, s)) if s else None for s in smi]
            if sum(x is not None for x in g) >= 2:
                zs, zg, zi = z([e[1] + 0.15 * e[3] for e in v]), z(g), z(ic)
                sp = {s: a + wg * b + wi * c for s, a, b, c in zip(smi, zs, zg, zi) if s}
                slots = [i for i, s in enumerate(L) if s in sp]
                if len(slots) >= 2:
                    items = sorted((L[i] for i in slots), key=lambda s: -sp[s])
                    L = list(L)
                    for i, s in zip(slots, items):
                        L[i] = s
        M, seen = [], set()
        for s in L + [s for s in d.split(";") if s]:
            if s not in seen:
                seen.add(s); M.append(s)
            if len(M) == 25:
                break
        lists[mid] = M
    return mrr(lists)



def main():
    global W, TRUTH, E2E, pool, direct, pp, name2smi, gl, ice, truth, es, K, tkey, sets
    W, TRUTH, E2E = sys.argv[1:4]


    pool = rd(os.path.join(W, "pool_candidates.csv"))
    direct = rd(os.path.join(W, "direct_submission.csv"))
    pp = {}
    for l in open(os.path.join(W, "pp_dump.txt"), encoding="utf-8"):
        p = l.rstrip("\n").split("\t")
        if len(p) >= 6:
            pp.setdefault(p[0], []).append((int(p[1]), float(p[2]), int(p[3]), float(p[4]), p[5]))
    need = {e[4] for v in pp.values() for e in v if e[2] == 1}
    name2smi = {}
    for l in open(os.path.join(W, "pool730_mols.txt"), encoding="utf-8"):
        t = l.find("\t")
        if t > 0 and l[:t] in need:
            name2smi[l[:t]] = l[t + 1:].rstrip("\r\n")
    gl, ice = {}, {}
    for l in open(os.path.join(W, "fw_out_gi.tsv"), encoding="utf-8"):
        p = l.rstrip("\n").split("\t")
        if len(p) >= 4:
            for d, x in ((gl, p[2]), (ice, p[3])):
                if x != "nan":
                    d[(p[0], p[1])] = float(x)
    print("candidates scored: gl %d ice %d" % (len(gl), len(ice)))
    truth = {}
    for l in open(TRUTH, encoding="utf-8"):
        t = l.rstrip("\r\n").split("\t")
        if len(t) >= 4:
            truth[t[0]] = t
    sys.path.insert(0, E2E)
    _argv = sys.argv
    sys.argv = [sys.argv[0]]
    import e2e_score as es  # importable module: Windows multiprocessing workers re-import it
    sys.argv = _argv
    allsmi = set(t[3] for t in truth.values())
    for v in list(pool.values()) + list(direct.values()):
        allsmi.update(s for s in v.split(";") if s)
    K = es.keys_for(sorted(allsmi))
    tkey = {m: (K[t[3]][0] or t[2]) for m, t in truth.items()}
    sets = sorted({t[1] for t in truth.values()})


    base, bper = run(0, 0)
    print("v40-equivalent (no forward model):", {k: round(v, 4) for k, v in base.items()})
    for wg, wi in ((0.5, 0), (0, 0.25), (0, 0.5), (0, 1.0), (0.25, 0.25), (0.5, 0.25), (0.5, 0.5), (0.5, 1.0), (1.0, 0.5),
                   (1.0, 1.0), (0.25, 0.5)):
        r, per = run(wg, wi)
        s = []
        for k in sets:
            d = [a - b for a, b in zip(per[k], bper[k])]
            mu = sum(d) / len(d)
            se = math.sqrt(sum((x - mu) ** 2 for x in d) / (len(d) - 1) / len(d))
            s.append(f"{k} {r[k]:.4f} ({r[k] - base[k]:+.4f} se {se:.4f})")
        print(f"gl {wg:<4} ice {wi:<4}: " + "  ".join(s), flush=True)



if __name__ == "__main__":
    main()
