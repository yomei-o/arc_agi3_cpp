r"""Morgan fingerprints for the whole candidate pool, bit-packed for C++.

Analog propagation needs a structural similarity between every pool candidate
and every spectrum-bearing library structure within a mass-shift window, which
is a lot of Tanimoto computations - cheap only if the fingerprints are packed
bits and popcount does the work, not Python loops over RDKit objects.

Radius 2, 1024 bits, folded (not sparse) so a fixed-width binary row works.
Multiprocessing because 729k molecules at a few ms each in RDKit is the
difference between 5 minutes and 40 on one core.

Layout: pool_fp.bin, one 128-byte (1024-bit) row per pool_mols.txt line, same
order. A molecule RDKit cannot parse gets an all-zero row (Tanimoto against
an all-zero row is defined as 0 here, never the best match).

    python export_fp.py <root>
"""
from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")
NBITS = 1024
_gen = None


def _init():
    global _gen
    _gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=NBITS)


def _fp_packed(smi: str) -> bytes:
    if not smi:
        return bytes(NBITS // 8)
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return bytes(NBITS // 8)
    bv = _gen.GetFingerprint(m)
    arr = np.zeros(NBITS, dtype=np.uint8)
    for i in bv.GetOnBits():
        arr[i] = 1
    return np.packbits(arr).tobytes()


def main() -> None:
    root = Path(sys.argv[1])
    smiles = []
    for line in (root / "pool_mols.txt").read_text(encoding="utf-8").splitlines():
        _, _, b = line.partition("\t")
        smiles.append(b)
    print(f"{len(smiles)} structures to fingerprint", flush=True)

    with Pool(16, initializer=_init) as p:
        rows = []
        for i, row in enumerate(p.imap(_fp_packed, smiles, chunksize=500)):
            rows.append(row)
            if i % 50000 == 0:
                print(f"  {i}/{len(smiles)}", end="\r", flush=True)
    with open(root / "pool_fp.bin", "wb") as f:
        f.write(b"".join(rows))
    print(f"\nwrote pool_fp.bin: {len(rows)} x {NBITS}-bit rows")


if __name__ == "__main__":
    main()
