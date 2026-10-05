r"""Merge train's own structures with COCONUT into one candidate pool.

Library search can only answer a molecule that already has a reference
spectrum (train's 275,810 structures). Everything else needs a candidate to
even be considered, structure-only, no spectrum required. This writes that
pool: every train structure plus every COCONUT structure not already in
train, deduplicated on InChIKey14 (the metric is stereochemistry-blind, so two
spellings of one skeleton would otherwise waste a slot), sorted by monoisotopic
mass for the same binary-search gate the spectral matcher already uses.

Mass is computed from the SMILES with RDKit rather than trusted from COCONUT's
own column, so train structures (which carry no precomputed mass) get one the
same way and the whole pool is internally consistent.

Layout (little-endian), mirroring head.bin/mols.txt:
  pool_head.bin   one record per candidate: mono_mass (f32), mol_id (i32),
                  has_spectrum (i32, 1 if it's one of train's 275,810)
  pool_mols.txt   name\tsmiles, indexed by mol_id

    python build_pool.py <root> <coconut_csv>
"""
from __future__ import annotations

import csv, struct, sys
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

RDLogger.DisableLog("rdApp.*")

root = Path(sys.argv[1])
coconut_path = Path(sys.argv[2])

names: list[str] = []
smiles: list[str] = []
seen: dict[str, int] = {}

n_train = 0
for line in (root / "train_mols.txt").read_text(encoding="utf-8").splitlines():
    a, _, b = line.partition("\t")
    if not a or a in seen:
        continue
    seen[a] = len(names)
    names.append(a)
    smiles.append(b)
    n_train += 1
print(f"train: {n_train} structures", flush=True)

csv.field_size_limit(10_000_000)
n_new = 0
with coconut_path.open(encoding="utf-8", errors="replace") as f:
    r = csv.DictReader(f)
    for row in r:
        ik = row.get("standard_inchi_key", "")
        if not ik:
            continue
        key = ik.split("-")[0]
        if key in seen:
            continue
        smi = row.get("canonical_smiles", "")
        if not smi:
            continue
        seen[key] = len(names)
        names.append(key)
        smiles.append(smi)
        n_new += 1
print(f"COCONUT: {n_new} new structures added, pool total {len(names)}", flush=True)

mass = [0.0] * len(names)
n_parsed = n_failed = 0
for i, smi in enumerate(smiles):
    if not smi:
        continue
    m = Chem.MolFromSmiles(smi)
    if m is None:
        n_failed += 1
        continue
    mass[i] = Descriptors.ExactMolWt(m)
    n_parsed += 1
    if i % 50000 == 0:
        print(f"  mass {i}/{len(names)}", end="\r", flush=True)
print(f"\nmass computed for {n_parsed}, RDKit could not parse {n_failed}")

order = sorted(range(len(names)), key=lambda i: mass[i])
with open(root / "pool_head.bin", "wb") as f:
    for i in order:
        hs = 1 if i < n_train else 0   # train entries were added first, in order
        f.write(struct.pack("<fii", mass[i], i, hs))
(root / "pool_mols.txt").write_text(
    "\n".join(f"{a}\t{b}" for a, b in zip(names, smiles)), encoding="utf-8")
print(f"wrote pool_head.bin ({len(names)} records, mass-sorted) and pool_mols.txt")
