r"""Turn the parquet into something C++ can mmap and scan.

Two and a half million spectra will be compared against four hundred, which is
a billion peak-list comparisons if done naively. Parquet and pandas are the
wrong shape for that: the work is a tight loop over float arrays, so the data
goes out as flat binary and the loop goes in C++.

Layout, all little-endian:
  head.bin   one record per spectrum, fixed width
  peaks.bin  the m/z and intensity pairs, concatenated, pointed into by head
  smiles.txt one line per molecule, indexed by head.mol
"""
from __future__ import annotations

import struct, sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else r"C:\prog\casmi")

ADDUCTS = ["[M+H]+", "[M-H]-", "[M+Na]+", "[M+NH4]+", "[M+K]+", "[M+Cl]-",
           "[M+CH2O2-H]-", "[M+H-H2O]+", "[2M+H]+", "[2M+Na]+"]
ADDUCT_ID = {a: i for i, a in enumerate(ADDUCTS)}

INSTR = ["timsTOF", "Orbitrap", "QTOF", "LC-ESI-QTOF", "ESI-QFT"]
INSTR_ID = {a: i for i, a in enumerate(INSTR)}


def export(src: Path, out: str, with_labels: bool) -> None:
    cols = ["precursor_mz", "adduct", "instrument_type",
            "ms2_mzs", "ms2_normalized_intensities"]
    cols += ["inchikey14", "normalized_smiles"] if with_labels else ["molecule_id"]

    f = pq.ParquetFile(src)
    head = open(ROOT / f"{out}_head.bin", "wb")
    peaks = open(ROOT / f"{out}_peaks.bin", "wb")
    mols: dict[str, int] = {}
    names: list[str] = []
    smiles: list[str] = []
    off = 0
    n = 0

    for batch in f.iter_batches(batch_size=100000, columns=cols):
        d = batch.to_pandas()
        for r in d.itertuples(index=False):
            mz = np.asarray(r.ms2_mzs, dtype=np.float32)
            it = np.asarray(r.ms2_normalized_intensities, dtype=np.float32)
            k = min(len(mz), len(it))
            if k == 0:
                continue
            mz, it = mz[:k], it[:k]
            o = np.argsort(mz)            # the comparison walks both lists in order
            peaks.write(mz[o].tobytes())
            peaks.write(it[o].tobytes())

            if with_labels:
                name = r.inchikey14 if isinstance(r.inchikey14, str) else ""
                smi = r.normalized_smiles if isinstance(r.normalized_smiles, str) else ""
            else:
                name, smi = r.molecule_id, ""
            if name not in mols:
                mols[name] = len(names)
                names.append(name)
                smiles.append(smi)
            elif smi and not smiles[mols[name]]:
                smiles[mols[name]] = smi

            head.write(struct.pack(
                "<fiiii",
                float(r.precursor_mz) if r.precursor_mz == r.precursor_mz else 0.0,
                mols[name], off, k,
                ADDUCT_ID.get(r.adduct, -1) * 16 + INSTR_ID.get(r.instrument_type, -1)))
            off += k
            n += 1
        print(f"  {n} spectra", end="\r", flush=True)

    head.close()
    peaks.close()
    (ROOT / f"{out}_mols.txt").write_text(
        "\n".join(f"{a}\t{b}" for a, b in zip(names, smiles)), encoding="utf-8")
    print(f"\n{out}: {n} spectra, {len(names)} molecules, {off} peaks")


export(ROOT / "train.parquet", "train", True)
export(ROOT / "test.parquet", "test", False)
