"""Shared library for several holdouts built with make_holdout.py --no-lib.

    python build_ylib.py <dir> <src_stem> <dst_stem> <denovo_qry,..> <sibling_qry,..> [xinstr_qry,..]

Starts from src (alllib = all of train, --floor 0.002). Every spectrum of a
molecule in a denovo query set is dropped; for sibling query sets only the
query spectra themselves are dropped (matched exactly on molecule, tag,
precursor and peak bytes), so the molecule's other spectra stay, as in the
production library. Models trained on dst never see a query spectrum or a
denovo molecule.
"""
import struct, sys
from collections import Counter
from pathlib import Path

d = Path(sys.argv[1]); src, dst = sys.argv[2], sys.argv[3]
denovo = [s for s in sys.argv[4].split(",") if s]
sibling = [s for s in (sys.argv[5].split(",") if len(sys.argv) > 5 else []) if s]
# xinstr sets (make_holdout --mode xinstr): every timsTOF spectrum (instrument id 0) of the
# molecule leaves the library; its spectra on other instruments stay
xinstr = [s for s in (sys.argv[6].split(",") if len(sys.argv) > 6 else []) if s]


def names(stem):
    return [l.rstrip("\r\n").split("\t")[0] for l in open(d / f"{stem}_mols.txt", encoding="utf-8")]


drop_mol = set()
for q in denovo:
    drop_mol |= set(names(q))
drop_spec = Counter()
for q in sibling:
    nm = names(q)
    hb = (d / f"{q}_head.bin").read_bytes()
    pk = (d / f"{q}_peaks.bin").read_bytes()
    for i in range(len(hb) // 20):
        p, mol, off, n, tag = struct.unpack_from("<fiiii", hb, 20 * i)
        drop_spec[(nm[mol], tag, n, struct.pack("<f", p), pk[8 * off:8 * (off + n)])] += 1
n_sib = sum(drop_spec.values())
drop_tims = set()
for q in xinstr:
    drop_tims |= set(names(q))
n_x = 0

sn = []
ss = []
for l in open(d / f"{src}_mols.txt", encoding="utf-8"):
    a, _, b = l.rstrip("\r\n").partition("\t")
    sn.append(a); ss.append(b)
hb = (d / f"{src}_head.bin").read_bytes()
peaks = open(d / f"{src}_peaks.bin", "rb")
oh, op = open(d / f"{dst}_head.bin", "wb"), open(d / f"{dst}_peaks.bin", "wb")
newid, nn, ns = {}, [], []
off = n_out = n_mol_drop = n_spec_drop = 0
for i in range(len(hb) // 20):
    p, mol, _, n, tag = struct.unpack_from("<fiiii", hb, 20 * i)
    blob = peaks.read(8 * n)
    k = sn[mol]
    if k in drop_mol:
        n_mol_drop += 1
        continue
    if k in drop_tims and tag % 16 == 0:
        n_x += 1
        continue
    key = (k, tag, n, struct.pack("<f", p), blob)
    if drop_spec.get(key, 0) > 0:
        drop_spec[key] -= 1
        n_spec_drop += 1
        continue
    if mol not in newid:
        newid[mol] = len(nn); nn.append(k); ns.append(ss[mol])
    op.write(blob)
    oh.write(struct.pack("<fiiii", p, newid[mol], off, n, tag))
    off += n; n_out += 1
oh.close(); op.close()
(d / f"{dst}_mols.txt").write_text("\n".join(f"{a}\t{b}" for a, b in zip(nn, ns)), encoding="utf-8")
print(f"{dst}: {n_out} spectra over {len(nn)} molecules; dropped {n_mol_drop} spectra of {len(drop_mol)} denovo "
      f"molecules, {n_spec_drop}/{n_sib} sibling query spectra (unmatched {sum(drop_spec.values())}), "
      f"{n_x} timsTOF spectra of {len(drop_tims)} xinstr molecules")
