r"""How much of the unreachable 84% does COCONUT actually reach?

Library search can only ever answer a molecule whose structure already has a
reference spectrum. The denovo holdout simulates the rest: 400 molecules with
every one of their spectra removed from the library, nothing to find by mass
or cosine no matter how clean the peaks are. This checks, before writing a
line of ranking code, how many of those answers exist in COCONUT at all - if
the ceiling is low, analog propagation has nothing to propagate from.

    python pool_recall.py <root> <coconut_csv>
"""
from __future__ import annotations

import csv, sys
from pathlib import Path

root = Path(sys.argv[1])
coconut_path = Path(sys.argv[2])

held = []
for line in (root / "dqry_mols.txt").read_text(encoding="utf-8").splitlines():
    a, _, b = line.partition("\t")
    if a:
        held.append((a, b))
print(f"held-out (denovo) molecules: {len(held)}")

held_keys = {k for k, _ in held}

csv.field_size_limit(10_000_000)
coco_keys = set()
n = 0
with coconut_path.open(encoding="utf-8", errors="replace") as f:
    r = csv.DictReader(f)
    for row in r:
        ik = row.get("standard_inchi_key", "")
        if ik:
            coco_keys.add(ik.split("-")[0])
        n += 1
        if n % 200000 == 0:
            print(f"  read {n}", end="\r", flush=True)
print(f"\nCOCONUT: {n} rows, {len(coco_keys)} unique InChIKey14")

hit = held_keys & coco_keys
print(f"\nheld-out molecules whose structure is in COCONUT: {len(hit)} / {len(held)}"
      f"  ({100.0*len(hit)/len(held):.1f}%)")

# And the library's own train set, for the overlap/novelty split.
train_keys = set()
for line in (root / "train_mols.txt").read_text(encoding="utf-8").splitlines():
    a = line.split("\t")[0]
    if a:
        train_keys.add(a)
print(f"train library: {len(train_keys)} unique InChIKey14")
new_from_coconut = coco_keys - train_keys
print(f"COCONUT structures NOT already in train: {len(new_from_coconut)}"
      f"  ({100.0*len(new_from_coconut)/len(coco_keys):.1f}% of COCONUT)")
