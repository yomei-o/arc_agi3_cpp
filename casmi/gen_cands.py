# gen_cands.py: generated candidates per query (analog + transformation), for
# the Phase 2 pool-augmentation evaluation.
#   python gen_cands.py <ad_dump.txt> <qry_mols.txt> <lib_mols.txt> <out.txt> [nproc] [maxrank] [cap]
# out: "G <query> <n>" then n lines
#   key14 \t smiles \t mass \t bestSim \t bestRank \t minSteps \t nSupport \t ops(;)
import sys
from multiprocessing import Pool
from gen_recall import *

def work2(args):
    qname, qmass, analogs, cap = args
    tol = max(0.003, 10e-6 * qmass)
    prod = {}
    for rank, (asmi, sim, nm) in enumerate(analogs):
        am = Chem.MolFromSmiles(asmi)
        if am is None:
            continue
        am = Chem.MolFromSmiles(Chem.MolToSmiles(am, isomericSmiles=False))
        if am is None:
            continue
        delta = qmass - Descriptors.ExactMolWt(am)
        if abs(delta) <= tol:
            continue
        for chain in ops_for_delta(delta, tol):
            cur = [am]
            for t in chain:
                nxt = {}
                for c in cur:
                    for p in apply(c, t):
                        k = key(p)
                        if k not in nxt:
                            nxt[k] = p
                    if len(nxt) > CAP:
                        break
                cur = list(nxt.values())
            for p in cur:
                k = key(p)
                e = prod.get(k)
                if e is None:
                    e = prod[k] = {'sim': 0.0, 'rank': 10**6, 'steps': 9, 'sup': set(), 'ops': set(), 'm': p}
                e['sim'] = max(e['sim'], sim)
                e['rank'] = min(e['rank'], rank)
                e['steps'] = min(e['steps'], len(chain))
                e['sup'].add(rank)
                e['ops'].add('/'.join(chain))
    items = sorted(prod.items(), key=lambda kv: (-kv[1]['sim'], kv[1]['steps'], kv[1]['rank']))[:cap]
    out = []
    seen = set()
    for k, e in items:
        try:
            ik = Chem.MolToInchiKey(e['m'])[:14]
        except Exception:
            continue
        if not ik or ik in seen:
            continue
        seen.add(ik)
        out.append(f"{ik}\t{k}\t{Descriptors.ExactMolWt(e['m']):.6f}\t{e['sim']:.4f}\t{e['rank']}\t{e['steps']}\t{len(e['sup'])}\t{';'.join(sorted(e['ops']))}")
    return qname, out

def main():
    dump, qmols, lmols, outp = sys.argv[1:5]
    nproc = int(sys.argv[5]) if len(sys.argv) > 5 else 8
    maxrank = int(sys.argv[6]) if len(sys.argv) > 6 else 80
    cap = int(sys.argv[7]) if len(sys.argv) > 7 else 1000
    lsmi = [l.rstrip('\r\n').split('\t')[1] if '\t' in l else '' for l in open(lmols, encoding='utf8')]
    tsmi = {}
    for l in open(qmols, encoding='utf8'):
        p = l.rstrip('\r\n').split('\t')
        tsmi[p[0]] = p[1] if len(p) > 1 else ''
    jobs = []
    for line in open(dump):
        p = line.split()
        if p[0] == 'Q':
            jobs.append([p[1], []])
        elif len(jobs[-1][1]) < maxrank:
            jobs[-1][1].append((lsmi[int(p[0])], float(p[1]), int(p[2])))
    args = []
    for name, an in jobs:
        tm = Chem.MolFromSmiles(tsmi.get(name, ''))
        if tm is None:
            continue
        args.append((name, Descriptors.ExactMolWt(tm), an, cap))
    with Pool(nproc) as pool, open(outp, 'w') as f:
        for i, (q, out) in enumerate(pool.imap_unordered(work2, args, chunksize=1)):
            f.write(f"G {q} {len(out)}\n")
            for o in out:
                f.write(o + '\n')
            if i % 50 == 0:
                print(i, flush=True)

if __name__ == '__main__':
    main()
