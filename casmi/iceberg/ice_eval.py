r"""ice_eval.py — glue between our v30 holdout runs and ICEBERG / GLACIER scores (iceberg_STATUS.md).

build <v30dir> <tag> <K> <outdir> [<mols.txt>]
    From the v30 run outputs <tag>_ins.csv (final list, names), <tag>_pp.txt (top-60 fused scores),
    <tag>_det.txt (PubChem SMILES) and f_<tag>.txt (every pool / generated candidate with SMILES), make the top-K
    list per query: the final 25 (with the inserted PubChem slots) then the next fused-list names up to K.
    Writes <outdir>\list_<tag>.json {mol: [[name, smiles, fused score | null, is_pubchem, formula], ...]} and
    <outdir>\cands_<tag>.json {mol: [smiles, ...]} (the ice_score.py input).

eval <outdir> <tag> <mols.txt> <q_<qry>.json> <msg_ik14 file> <score json>[,<score json>...] [--seed 1]
    Re-ranks the list with forward-model scores and writes one CSV per variant (<outdir>\out_<tag>_<variant>.csv,
    same format as the run's _ins.csv) + per-query reciprocal ranks (rr_<tag>.tsv). Variants:
      base                      the K-list cut to 25 (= the v30 final list)
      fgr_pure                  same-formula groups (>= 2 members in the K-list) re-ordered by the model score alone,
                                group slots kept (the public pipelines' slot-preserving re-rank)
      fgr_l1                    same, key z(s_list) + 1.0 z(model), s_list = 1/(3 + rank) (the public RRF-style score)
      add_w{0,0.25,0.5,1}       every non-PubChem item re-sorted by z(fused) + w z(model) over the K-list items with a
                                fused score (PubChem slots keep their positions); add_w0 = fused re-sort alone, the
                                reference for the add_* rows (pool-popularity / promotion moves are not in the pp score)
      *_mh                      the same with the model score from [M+H]+ spectra only
      *_rand                    control: the model score replaced by uniform noise (same coverage)
    Model score of a candidate = mean entropy similarity (or cosine with --sim cos) over its scored spectra;
    candidates / queries without a score get z = 0 (no move). --ceknown: only spectra with a recorded CE.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

PPOP = 0.15


def formula(smi):
    from rdkit import Chem
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return "?" + smi
    return CalcMolFormula(m)


def build(v30, tag, K, outdir, mols=None):
    v30, outdir = Path(v30), Path(outdir)
    truth_smi = {}
    if mols:
        for l in open(mols, encoding="utf-8"):
            t = l.rstrip("\n").split("\t")
            if len(t) >= 2:
                truth_smi[t[0]] = t[1]
    smi = {}
    libmax = {}
    cur = None
    for line in open(v30 / f"f_{tag}.txt", encoding="utf-8"):
        t = line.split()
        if t[0] == "Q":
            cur = smi.setdefault(t[1], {})
            qn = t[1]
            libmax[qn] = 0.0
            continue
        libmax[qn] = max(libmax[qn], min(1.0, float(t[3])))   # best library similarity of the query
        name = t[6]
        if name.startswith("GEN") and name[3:].isdigit():   # generated candidate GEN<k> = G<k> in the csv
            name = "G" + name[3:]
        cur[name] = t[-1]
    pc = {}
    q = None
    for line in open(v30 / f"{tag}_det.txt", encoding="utf-8"):
        t = line.rstrip("\n").split("\t")
        if t[0] == "Q":
            q = t[1]
            continue
        if t[0] == "V2" or len(t) < 3:
            continue
        pc.setdefault(q, {})["P" + t[0]] = t[-1]
    fused = {}
    order = {}
    for line in open(v30 / f"{tag}_pp.txt", encoding="utf-8"):
        t = line.rstrip("\n").split("\t")
        # pp dump: mol, rank, fused score, has-pop flag, log popularity, name. The final pool order is
        # fused + PPOP * pop (ppop=0.15, phys_v28pc pool popularity) -> store that as the list's fused score.
        fused.setdefault(t[0], {})[t[5]] = float(t[2]) + PPOP * float(t[4])
        order.setdefault(t[0], []).append(t[5])
    L, C = {}, {}
    miss = 0
    with open(v30 / f"{tag}_ins.csv", encoding="utf-8") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            mid = row[0]
            names = row[1].split(";") if len(row) > 1 and row[1] else []
            for n in order.get(mid, []):
                if len(names) >= K:
                    break
                if n not in names:
                    names.append(n)
            items = []
            for n in names[:K]:
                isp = n[:1] == "P" and n[1:].isdigit()   # PubChem names are P<cid>; IK14 names may start with P
                s = pc.get(mid, {}).get(n) if isp else smi.get(mid, {}).get(n)
                if s is None and n == mid and truth_smi:
                    # rm=1 runs: a PubChem slot that IS the answer is written under the answer's IK14 (eval name),
                    # not P<cid>; its structure is the answer's
                    s, isp = truth_smi.get(n), True
                if s is None:
                    miss += 1
                    if miss <= 8:
                        print("  no SMILES:", mid, n)
                    continue
                items.append([n, s, fused.get(mid, {}).get(n), int(isp), formula(s)])
            L[mid] = items
            C[mid] = [it[1] for it in items]
    json.dump(L, open(outdir / f"list_{tag}.json", "w"))
    json.dump(C, open(outdir / f"cands_{tag}.json", "w"))
    json.dump(libmax, open(outdir / f"lib_{tag}.json", "w"))
    n = sum(len(v) for v in L.values())
    print(f"{tag}: {len(L)} queries, {n} items ({n / max(1, len(L)):.1f}/query), names without SMILES {miss}")


def z(x):
    x = np.asarray(x, float)
    sd = x.std()
    return (x - x.mean()) / sd if sd > 1e-12 else np.zeros_like(x)


def model_score(S, Q, mid, smi, mh_only, ceknown, sim):
    rows = S.get(mid, {}).get(smi)
    if not rows:
        return None
    k = 0 if sim == "ent" else 1
    v = []
    for e in rows:
        s = Q[mid][e[3]]
        if mh_only and s["adduct"] != "[M+H]+":
            continue
        if ceknown and not e[2]:
            continue
        v.append(e[k])
    return float(np.mean(v)) if v else None


def rerank(items, ms, variant):
    """items: list entries; ms: model score per item (None = missing). returns names in the new order."""
    n = len(items)
    names = [it[0] for it in items]
    have = [m is not None for m in ms]
    mz = np.zeros(n)
    if any(have):
        idx = [i for i in range(n) if have[i]]
        mz[idx] = z([ms[i] for i in idx])
    if variant.startswith("fgr"):
        out = list(names)
        groups = {}
        for i, it in enumerate(items):
            groups.setdefault(it[4], []).append(i)
        sl = z([1.0 / (3 + i) for i in range(n)])
        for g in groups.values():
            if len(g) < 2 or sum(have[i] for i in g) < 2:
                continue
            if variant == "fgr_pure":
                key = {i: (mz[i] if have[i] else -1e9, -i) for i in g}   # unscored members sink inside the group
            else:
                key = {i: (sl[i] + 1.0 * mz[i], -i) for i in g}
            new = sorted(g, key=lambda i: key[i], reverse=True)
            for slot, i in zip(g, new):
                out[slot] = names[i]
        return out
    w = float(variant.split("_w")[1])
    fixed = {i for i, it in enumerate(items) if it[3] or it[2] is None}
    free = [i for i in range(n) if i not in fixed]
    if len(free) < 2:
        return names
    fz = z([items[i][2] for i in free])
    key = {i: fz[j] + w * mz[i] for j, i in enumerate(free)}
    new = sorted(free, key=lambda i: (key[i], -i), reverse=True)
    out = list(names)
    for slot, i in zip(free, new):
        out[slot] = names[i]
    return out


GATE = 0.9   # the public pipelines' library gate (lib_max < 0.9) for the PubChem channel


def rerank_v(items, ms, v, libmax):
    """variants on top of rerank(): gate_* = only when the query's best library similarity < GATE;
    comb_wX = add_wX, then the slot-preserving same-formula re-rank fgr_l1 on the new order."""
    if v.startswith("gate_"):
        if libmax >= GATE:
            return [it[0] for it in items]
        v = v[5:]
    if v.startswith("comb_"):
        first = rerank(items, ms, "add_" + v[5:])
        pos = {it[0]: i for i, it in enumerate(items)}
        it2 = [items[pos[n]] for n in first]
        ms2 = [ms[pos[n]] for n in first]
        return rerank(it2, ms2, "fgr_l1")
    return rerank(items, ms, v)


def evaluate(outdir, tag, mols, qjson, msgfile, scorefiles, seed=1, sim="ent", ceknown=False):
    outdir = Path(outdir)
    L = json.load(open(outdir / f"list_{tag}.json"))
    Q = json.load(open(qjson))
    S = {}
    for f in scorefiles.split(","):
        for mid, d in json.load(open(f)).items():
            S.setdefault(mid, {}).update(d)
    truth = [l.split("\t")[0] for l in open(mols, encoding="utf-8") if l.strip()]
    msg = set(k for f in msgfile.split(",") for k in Path(f).read_text().split())
    rng = np.random.default_rng(seed)
    variants = ["base", "fgr_pure", "fgr_l1", "add_w0", "add_w0.25", "add_w0.5", "add_w1"]
    variants += ["gate_add_w0.5", "gate_add_w1", "comb_w0.5", "gate_comb_w0.5"]
    allv = variants + [v + "_mh" for v in variants[1:]] + [v + "_rand" for v in variants[1:]]
    libf = outdir / f"lib_{tag}.json"
    LIB = json.load(open(libf)) if libf.exists() else {}
    lists = {v: {} for v in allv}
    cover = dict(queries=0, scored_queries=0, items=0, scored_items=0, scored_items_mh=0)
    for mid, items in L.items():
        cover["queries"] += 1
        ms = [model_score(S, Q, mid, it[1], False, ceknown, sim) for it in items]
        mh = [model_score(S, Q, mid, it[1], True, ceknown, sim) for it in items]
        rnd = [float(rng.random()) if m is not None else None for m in ms]
        cover["items"] += len(items)
        cover["scored_items"] += sum(m is not None for m in ms)
        cover["scored_items_mh"] += sum(m is not None for m in mh)
        cover["scored_queries"] += int(any(m is not None for m in ms))
        lists["base"][mid] = [it[0] for it in items]
        for v in variants[1:]:
            for suf, sc in (("", ms), ("_mh", mh), ("_rand", rnd)):
                lists[v + suf][mid] = rerank_v(items, sc, v, LIB.get(mid, 0.0))
    rr = {}
    for v in allv:
        p = outdir / f"out_{tag}_{v}.csv"
        with open(p, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["molecule_id", "smiles"])
            for mid, names in lists[v].items():
                w.writerow([mid, ";".join(names[:25])])
        rr[v] = np.array([1.0 / (lists[v][k][:25].index(k) + 1) if k in lists[v].get(k, [])[:25] else 0.0
                          for k in truth])
    inmsg = np.array([k in msg for k in truth])
    with open(outdir / f"rr_{tag}.tsv", "w") as f:
        f.write("mol\tin_msg\t" + "\t".join(allv) + "\n")
        for i, k in enumerate(truth):
            f.write(f"{k}\t{int(inmsg[i])}\t" + "\t".join(f"{rr[v][i]:.5f}" for v in allv) + "\n")
    print(f"{tag}: coverage {cover}; answers in MassSpecGym {inmsg.sum()}/{len(truth)}")
    B = 2000
    bi = np.random.default_rng(7).integers(0, len(truth), (B, len(truth)))
    print(f"{'variant':16s} {'MRR':>6s} {'d':>7s} {'95% CI':>17s} | {'MSG d':>7s} {'nonMSG d':>8s} [95% CI]")
    for v in allv:
        d = rr[v] - rr["base"]
        bs = d[bi].mean(1)
        dn = d[~inmsg]
        bn = dn[np.random.default_rng(8).integers(0, len(dn), (B, len(dn)))].mean(1) if len(dn) else np.zeros(1)
        print(f"{v:16s} {rr[v].mean():.4f} {d.mean():+.4f} [{np.percentile(bs, 2.5):+.4f},{np.percentile(bs, 97.5):+.4f}]"
              f" | {d[inmsg].mean() if inmsg.any() else 0:+.4f} {dn.mean() if len(dn) else 0:+.4f}"
              f" [{np.percentile(bn, 2.5):+.4f},{np.percentile(bn, 97.5):+.4f}]")


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5], sys.argv[6] if len(sys.argv) > 6 else None)
    elif sys.argv[1] == "eval":
        opts = sys.argv[8:]
        evaluate(*sys.argv[2:8], seed=1, sim="cos" if "--cos" in opts else "ent", ceknown="--ceknown" in opts)
