"""pc_det_rank.py <det.txt> [...]: truth's rank inside the PubChem list of a phys_v2Xpc pcdet dump.

Per query block (line 'Q ...', then 'V2 ...', then one line per listed PubChem candidate: cid, truth flag, ...),
prints: queries, truth listed (count), MRR@25 of the PubChem list alone (1/rank if rank<=25), top1, top10.
"""
import sys

for path in sys.argv[1:]:
    nq = listed = 0
    rr = top1 = top10 = 0.0
    rank = -1
    pos = 0

    def flush():
        global listed, rr, top1, top10
        if rank >= 0:
            listed += 1
            if rank < 25:
                rr += 1.0 / (rank + 1)
            top1 += rank == 0
            top10 += rank < 10

    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            t = line.rstrip('\n').split('\t')
            if t[0] == 'Q':
                if nq:
                    flush()
                nq += 1
                rank = -1
                pos = 0
            elif t[0] == 'V2':
                continue
            elif len(t) > 2 and t[0].lstrip('-').isdigit():
                if t[1] == '1' and rank < 0:
                    rank = pos
                pos += 1
    if nq:
        flush()
    print(f"{path}: queries {nq}, truth listed {listed}, PubChem-list MRR@25 {rr / max(1, nq):.4f}, top1 {int(top1)}, top10 {int(top10)}")
