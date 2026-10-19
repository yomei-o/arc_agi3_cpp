"""Query set of training molecules for within-window decoys (fp3_STATUS.md).

    python fp3_dq.py <dir> <lib_stem> <out_stem> <n_mols> [seed]

Picks n_mols random molecules of <lib_stem> (those with at least one spectrum),
writes <out_stem>_head.bin (the molecule's first spectrum, mol re-indexed,
peaks not needed by win_build: n kept, off 0) and <out_stem>_mols.txt
(name \t SMILES, same lines as the library). win_build <dir> <out_stem> ...
sampleK=64 then gives ~64 random PubChem isomers (same +-10 ppm window) each.
"""
import random, struct, sys
from pathlib import Path

d = Path(sys.argv[1]); lib, out = sys.argv[2], sys.argv[3]
n = int(sys.argv[4]); seed = int(sys.argv[5]) if len(sys.argv) > 5 else 7
mols = [l.rstrip("\r\n") for l in open(d / f"{lib}_mols.txt", encoding="utf-8")]
hb = (d / f"{lib}_head.bin").read_bytes()
first = {}
for i in range(len(hb) // 20):
    p, mol, off, cnt, tag = struct.unpack_from("<fiiii", hb, 20 * i)
    if cnt > 0 and mol not in first:
        first[mol] = (p, cnt, tag)
cand = sorted(first)
random.Random(seed).shuffle(cand)
pick = sorted(cand[:n])
with open(d / f"{out}_head.bin", "wb") as oh, open(d / f"{out}_mols.txt", "w", encoding="utf-8", newline="\n") as om:
    for j, m in enumerate(pick):
        p, cnt, tag = first[m]
        oh.write(struct.pack("<fiiii", p, j, 0, cnt, tag))
        om.write(mols[m] + "\n")
print(f"{out}: {len(pick)} molecules of {len(cand)} from {lib}")
