# gen_recall.py: Phase 1 - how often does editing a spectral analog by a known
# transformation reproduce the true structure?
#   python gen_recall.py <ad_dump.txt> <qry_mols.txt> <lib_mols.txt> <out.jsonl> [nproc] [maxrank]
import sys, json, itertools
from multiprocessing import Pool
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
RDLogger.DisableLog('rdApp.*')

H = 1.00782503207
# substituent groups attached at an X-H site (atom 0 attaches); delta = group - H
GROUPS = {
    'OH':      ('O',                        ('C',)),          # hydroxylation
    'Me':      ('C',                        ('O', 'N', 'S', 'Car', 'C')),
    'Ac':      ('C(C)=O',                   ('O', 'N')),
    'Hex':     ('C1OC(CO)C(O)C(O)C1O',      ('O', 'Car')),
    'dHex':    ('C1OC(C)C(O)C(O)C1O',       ('O',)),
    'Pen':     ('C1OCC(O)C(O)C1O',          ('O',)),
    'PenF':    ('C1OC(CO)C(O)C1O',          ('O',)),
    'Api':     ('C1OCC(O)(CO)C1O',          ('O',)),
    'GlcA':    ('C1OC(C(=O)O)C(O)C(O)C1O',  ('O',)),
    'Mal':     ('C(=O)CC(=O)O',             ('O',)),
    'SO3':     ('S(=O)(=O)O',               ('O',)),
    'Prenyl':  ('CC=C(C)C',                 ('Car', 'O')),
}
GMOL = {k: Chem.MolFromSmiles(v[0]) for k, v in GROUPS.items()}
GDELTA = {k: Descriptors.ExactMolWt(GMOL[k]) - 2 * H for k in GROUPS}  # group-H minus replaced H
# transformation types with their mass deltas (signed)
T = {}
for k, d in GDELTA.items():
    T['+' + k] = d
    T['-' + k] = -d
T['+NO'] = 15.9949146   # N/S oxide
T['+H2'] = 2 * H
T['-H2'] = -2 * H
T['+H2O'] = 18.0105647
T['-H2O'] = -18.0105647
T['-NO'] = -15.9949146
# '+OH' and '+NO' share a mass; '-OH','-NO' too: handled by the same Δ match
SINGLE = sorted(T)

def sites(m, kinds):
    out = []
    for a in m.GetAtoms():
        if a.GetTotalNumHs() == 0 or a.GetFormalCharge() != 0:
            continue
        s = a.GetSymbol()
        if s == 'C':
            if (a.GetIsAromatic() and 'Car' in kinds) or (not a.GetIsAromatic() and 'C' in kinds) or \
               (a.GetIsAromatic() and 'C' in kinds and 'Car' not in kinds):
                out.append(a.GetIdx())
        elif s in kinds:
            out.append(a.GetIdx())
    return out

def addH(a, d):
    if a.GetNoImplicit() or a.GetNumExplicitHs() > 0:
        a.SetNumExplicitHs(max(0, a.GetNumExplicitHs() + d))

def finish(rw):
    try:
        m = rw.GetMol()
        Chem.SanitizeMol(m)
        return m
    except Exception:
        return None

def attach(m, gk):
    g = GMOL[gk]
    res = []
    for i in sites(m, GROUPS[gk][1]):
        rw = Chem.RWMol(Chem.CombineMols(m, g))
        addH(rw.GetAtomWithIdx(i), -1)
        rw.AddBond(i, m.GetNumAtoms(), Chem.BondType.SINGLE)
        p = finish(rw)
        if p is not None:
            res.append(p)
    return res

def noxide(m):
    res = []
    for a in m.GetAtoms():
        if a.GetFormalCharge() != 0 or a.GetTotalNumHs() != 0:
            continue
        if a.GetSymbol() == 'N' and a.GetDegree() == 3 and not a.GetIsAromatic() and \
                all(b.GetBondType() == Chem.BondType.SINGLE for b in a.GetBonds()) or \
           a.GetSymbol() == 'N' and a.GetIsAromatic() and a.GetDegree() == 2:
            rw = Chem.RWMol(m)
            o = rw.AddAtom(Chem.Atom(8))
            rw.GetAtomWithIdx(a.GetIdx()).SetFormalCharge(1)
            rw.GetAtomWithIdx(o).SetFormalCharge(-1)
            rw.AddBond(a.GetIdx(), o, Chem.BondType.SINGLE)
            p = finish(rw)
            if p is not None:
                res.append(p)
        elif a.GetSymbol() == 'S' and a.GetDegree() == 2 and not a.GetIsAromatic():
            rw = Chem.RWMol(m)
            o = rw.AddAtom(Chem.Atom(8))
            rw.AddBond(a.GetIdx(), o, Chem.BondType.DOUBLE)
            p = finish(rw)
            if p is not None:
                res.append(p)
    return res

def de_noxide(m):
    res = []
    for b in m.GetBonds():
        x, y = b.GetBeginAtom(), b.GetEndAtom()
        for n, o in ((x, y), (y, x)):
            if n.GetSymbol() == 'N' and n.GetFormalCharge() == 1 and o.GetSymbol() == 'O' and \
                    o.GetFormalCharge() == -1 and o.GetDegree() == 1:
                rw = Chem.RWMol(m)
                rw.GetAtomWithIdx(n.GetIdx()).SetFormalCharge(0)
                rw.RemoveAtom(o.GetIdx())
                p = finish(rw)
                if p is not None:
                    res.append(p)
    return res

def frag_atoms(m, start, banned_bond):
    seen = {start}
    st = [start]
    while st:
        a = st.pop()
        for b in m.GetAtomWithIdx(a).GetBonds():
            if b.GetIdx() == banned_bond:
                continue
            o = b.GetOtherAtomIdx(a)
            if o not in seen:
                seen.add(o)
                st.append(o)
    return seen

def removals(m):
    """cut each acyclic single bond, drop one side, cap with H: [(delta, mol)]"""
    res = []
    n = m.GetNumAtoms()
    for b in m.GetBonds():
        if b.IsInRing() or b.GetBondType() != Chem.BondType.SINGLE:
            continue
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        for keep, drop in ((i, j), (j, i)):
            side = frag_atoms(m, drop, b.GetIdx())
            if len(side) > 16 or len(side) >= n - 1:
                continue
            if any(m.GetAtomWithIdx(k).GetFormalCharge() for k in side):
                continue
            rw = Chem.RWMol(m)
            addH(rw.GetAtomWithIdx(keep), 1)
            rw.RemoveBond(i, j)
            for k in sorted(side, reverse=True):
                rw.RemoveAtom(k)
            p = finish(rw)
            if p is not None:
                res.append(p)
    return res

def hydro(m):  # +H2: saturate one non-aromatic double bond
    res = []
    for b in m.GetBonds():
        if b.GetBondType() == Chem.BondType.DOUBLE and not b.GetIsAromatic():
            if b.GetBeginAtom().GetFormalCharge() or b.GetEndAtom().GetFormalCharge():
                continue
            if b.GetBeginAtom().GetSymbol() == 'S' or b.GetEndAtom().GetSymbol() == 'S':
                continue
            rw = Chem.RWMol(m)
            rw.GetBondWithIdx(b.GetIdx()).SetBondType(Chem.BondType.SINGLE)
            addH(rw.GetAtomWithIdx(b.GetBeginAtomIdx()), 1)
            addH(rw.GetAtomWithIdx(b.GetEndAtomIdx()), 1)
            p = finish(rw)
            if p is not None:
                res.append(p)
    return res

def dehydro(m):  # -H2: new double bond between two adjacent non-aromatic H-bearing atoms
    res = []
    for b in m.GetBonds():
        if b.GetBondType() != Chem.BondType.SINGLE or b.GetIsAromatic():
            continue
        x, y = b.GetBeginAtom(), b.GetEndAtom()
        if x.GetIsAromatic() or y.GetIsAromatic() or x.GetTotalNumHs() == 0 or y.GetTotalNumHs() == 0:
            continue
        if x.GetFormalCharge() or y.GetFormalCharge():
            continue
        if x.GetSymbol() not in 'CNO' or y.GetSymbol() not in 'CNO':
            continue
        if x.GetSymbol() != 'C' and y.GetSymbol() != 'C':
            continue
        rw = Chem.RWMol(m)
        for a in (x, y):
            addH(rw.GetAtomWithIdx(a.GetIdx()), -1)
        rw.GetBondWithIdx(b.GetIdx()).SetBondType(Chem.BondType.DOUBLE)
        p = finish(rw)
        if p is not None:
            res.append(p)
    return res

def hydrate(m):  # +H2O: add water across C=C, or hydrolyse a ring C-O/C-N bond
    res = []
    for b in m.GetBonds():
        x, y = b.GetBeginAtom(), b.GetEndAtom()
        if b.GetBondType() == Chem.BondType.DOUBLE and not b.GetIsAromatic() and \
                x.GetSymbol() == 'C' and y.GetSymbol() == 'C':
            for c in (x, y):
                rw = Chem.RWMol(m)
                rw.GetBondWithIdx(b.GetIdx()).SetBondType(Chem.BondType.SINGLE)
                o = rw.AddAtom(Chem.Atom(8))
                rw.AddBond(c.GetIdx(), o, Chem.BondType.SINGLE)
                addH(rw.GetAtomWithIdx((y if c is x else x).GetIdx()), 1)
                p = finish(rw)
                if p is not None:
                    res.append(p)
        elif b.IsInRing() and b.GetBondType() == Chem.BondType.SINGLE and not b.GetIsAromatic():
            for c, h in ((x, y), (y, x)):
                if c.GetSymbol() == 'C' and h.GetSymbol() in 'ON' and not h.GetIsAromatic():
                    rw = Chem.RWMol(m)
                    rw.RemoveBond(x.GetIdx(), y.GetIdx())
                    o = rw.AddAtom(Chem.Atom(8))
                    rw.AddBond(c.GetIdx(), o, Chem.BondType.SINGLE)
                    addH(rw.GetAtomWithIdx(h.GetIdx()), 1)
                    p = finish(rw)
                    if p is not None:
                        res.append(p)
    return res

def dehydrate(m):  # -H2O: C(OH)-C(H) -> C=C, or ring closure OH + X-H -> C-X (5-7 ring)
    res = []
    dm = None
    for o in m.GetAtoms():
        if o.GetSymbol() != 'O' or o.GetDegree() != 1 or o.GetTotalNumHs() != 1:
            continue
        c = o.GetNeighbors()[0]
        if c.GetSymbol() != 'C' or c.GetIsAromatic():
            continue
        cb = m.GetBondBetweenAtoms(o.GetIdx(), c.GetIdx())
        if cb.GetBondType() != Chem.BondType.SINGLE:
            continue
        # elimination
        for nb in c.GetNeighbors():
            if nb.GetIdx() == o.GetIdx() or nb.GetSymbol() != 'C' or nb.GetIsAromatic() or nb.GetTotalNumHs() == 0:
                continue
            bb = m.GetBondBetweenAtoms(c.GetIdx(), nb.GetIdx())
            if bb.GetBondType() != Chem.BondType.SINGLE:
                continue
            rw = Chem.RWMol(m)
            rw.GetBondWithIdx(bb.GetIdx()).SetBondType(Chem.BondType.DOUBLE)
            addH(rw.GetAtomWithIdx(nb.GetIdx()), -1)
            rw.RemoveAtom(o.GetIdx())
            p = finish(rw)
            if p is not None:
                res.append(p)
        # cyclisation
        if dm is None:
            dm = Chem.GetDistanceMatrix(m)
        for x in m.GetAtoms():
            if x.GetSymbol() not in 'ON' or x.GetTotalNumHs() == 0 or x.GetIdx() == o.GetIdx() or x.GetIsAromatic():
                continue
            d = dm[c.GetIdx()][x.GetIdx()]
            if 4 <= d <= 6:
                rw = Chem.RWMol(m)
                addH(rw.GetAtomWithIdx(x.GetIdx()), -1)
                rw.AddBond(c.GetIdx(), x.GetIdx(), Chem.BondType.SINGLE)
                rw.RemoveAtom(o.GetIdx())
                p = finish(rw)
                if p is not None:
                    res.append(p)
    return res

_rem_cache = {}
def apply(m, t):
    if t == '+NO':
        return noxide(m)
    if t == '-NO':
        return de_noxide(m)
    if t == '+H2':
        return hydro(m)
    if t == '-H2':
        return dehydro(m)
    if t == '+H2O':
        return hydrate(m)
    if t == '-H2O':
        return dehydrate(m)
    if t[0] == '+':
        return attach(m, t[1:])
    # removal of a group: generic cut, keep products whose mass drop matches
    mw = Descriptors.ExactMolWt(m)
    want = T[t]
    return [p for p in removals(m) if abs(Descriptors.ExactMolWt(p) - mw - want) < 0.002]

def key(m):
    return Chem.MolToSmiles(m, isomericSmiles=False)

def ops_for_delta(delta, tol):
    out = []
    for t in SINGLE:
        if abs(T[t] - delta) <= tol:
            out.append((t,))
    for t1, t2 in itertools.combinations_with_replacement(SINGLE, 2):
        if t1[1:] == t2[1:] and t1[0] != t2[0]:
            continue  # +X-X
        if abs(T[t1] + T[t2] - delta) <= tol:
            out.append((t1, t2))
    return out

CAP = 2000

def work(args):
    qname, tsmi, qmass, analogs = args
    tm = Chem.MolFromSmiles(tsmi)
    tform = rdMolDescriptors.CalcMolFormula(tm) if tm else None
    tkey = key(tm) if tm else None
    tol = max(0.003, 10e-6 * qmass)
    prod = {}  # smiles -> [minrank1, minrank2, best sim, ops set]
    hits = []
    iso_hit = None
    nmatch_an = 0
    for rank, (asmi, sim, nm) in enumerate(analogs):
        am = Chem.MolFromSmiles(asmi)
        if am is None:
            continue
        am = Chem.MolFromSmiles(Chem.MolToSmiles(am, isomericSmiles=False))  # drop stereo: implicit Hs
        if am is None:
            continue
        amass = Descriptors.ExactMolWt(am)
        delta = qmass - amass
        if abs(delta) <= tol:
            if key(am) == tkey:
                iso_hit = rank if iso_hit is None else iso_hit
            continue
        ops = ops_for_delta(delta, tol)
        if not ops:
            continue
        nmatch_an += 1
        for chain in ops:
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
                step = len(chain)
                if e is None:
                    e = prod[k] = [10**6, 10**6, 0.0, set(), p]
                if step == 1:
                    e[0] = min(e[0], rank)
                else:
                    e[1] = min(e[1], rank)
                e[2] = max(e[2], sim)
                e[3].add('/'.join(chain))
    # truth check: canonical smiles, then InChIKey14 for formula-equal products
    tk14 = qname
    for k, e in prod.items():
        hit = (k == tkey)
        if not hit and tform and rdMolDescriptors.CalcMolFormula(e[4]) == tform:
            try:
                hit = Chem.MolToInchiKey(e[4])[:14] == tk14
            except Exception:
                hit = False
        if hit:
            # position in a naive order: best generating-analog sim desc, one-step first
            mine = (-e[2], min(e[0], e[1]) if e[0] < 10**6 else 10**6 + e[1])
            pos = sum(1 for f in prod.values() if (-f[2], f[0] if f[0] < 10**6 else 10**6 + f[1]) < mine)
            hits.append({'r1': e[0], 'r2': e[1], 'sim': e[2], 'ops': sorted(e[3]), 'pos': pos})
    # candidate counts by rank cutoff and step
    cnt = {}
    for K in (10, 20, 40, 80, 200):
        cnt[K] = [sum(1 for e in prod.values() if e[0] < K), sum(1 for e in prod.values() if min(e[0], e[1]) < K)]
    return {'q': qname, 'nan': len(analogs), 'nmatch_an': nmatch_an, 'iso_hit': iso_hit,
            'hits': hits, 'cnt': cnt, 'top_sim': analogs[0][1] if analogs else 0.0}

def main():
    dump, qmols, lmols, outp = sys.argv[1:5]
    nproc = int(sys.argv[5]) if len(sys.argv) > 5 else 8
    maxrank = int(sys.argv[6]) if len(sys.argv) > 6 else 80
    lsmi = []
    for line in open(lmols, encoding='utf8'):
        p = line.rstrip('\r\n').split('\t')
        lsmi.append(p[1] if len(p) > 1 else '')
    tsmi = {}
    for line in open(qmols, encoding='utf8'):
        p = line.rstrip('\r\n').split('\t')
        tsmi[p[0]] = p[1] if len(p) > 1 else ''
    jobs = []
    cur = None
    for line in open(dump):
        p = line.split()
        if p[0] == 'Q':
            cur = [p[1], []]
            jobs.append(cur)
        else:
            if len(cur[1]) < maxrank:
                cur[1].append((int(p[0]), float(p[1]), int(p[2])))
    args = []
    for name, an in jobs:
        tm = Chem.MolFromSmiles(tsmi.get(name, ''))
        if tm is None:
            continue
        # query neutral mass: the true structure's exact mass (the test gives
        # precursor + adduct; their error is within the 10 ppm tolerance)
        args.append((name, tsmi[name], Descriptors.ExactMolWt(tm), [(lsmi[a], s, n) for a, s, n in an]))
    with Pool(nproc) as pool, open(outp, 'w') as f:
        for i, r in enumerate(pool.imap_unordered(work, args, chunksize=1)):
            f.write(json.dumps(r) + '\n')
            f.flush()
            if i % 20 == 0:
                print(i, flush=True)

if __name__ == '__main__':
    main()
