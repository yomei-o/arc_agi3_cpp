r"""Carve a test set out of the training data, shaped like the real one.

The real test is 400 molecules, 1,213 spectra, every one of them timsTOF, with
precursor masses between 245 and 460. A holdout that does not look like that
would measure the wrong thing - and on the last project a proxy that did not
match the real conditions invalidated nine days of comparisons, so this one is
built to match.

One spectrum of each held-out molecule becomes the query and the REST stay in
the library. Removing the molecule entirely was the first attempt and it scored
zero by construction - there was nothing left to find. The real test molecules
are not strangers either: all four hundred of their precursor masses exist in
the training set, so the task is to pick the right molecule out of about a
hundred with the same mass, not to invent one that is not there. The query
spectrum itself is excluded, so nothing is found by matching itself.

    python make_holdout.py [root]
"""
from __future__ import annotations

import struct, sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else r"C:\prog\casmi")
N_MOL = 400
MZ_LO, MZ_HI = 245.0, 461.0
RNG = np.random.default_rng(12345)

ADDUCTS = ["[M+H]+", "[M-H]-", "[M+Na]+", "[M+NH4]+", "[M+K]+", "[M+Cl]-",
           "[M+CH2O2-H]-", "[M+H-H2O]+", "[2M+H]+", "[2M+Na]+"]
ADDUCT_ID = {a: i for i, a in enumerate(ADDUCTS)}
INSTR = ["timsTOF", "Orbitrap", "QTOF", "LC-ESI-QTOF", "ESI-QFT"]
INSTR_ID = {a: i for i, a in enumerate(INSTR)}


def main() -> None:
    f = pq.ParquetFile(ROOT / "train.parquet")

    # First pass: which molecules could stand in for a test molecule?
    seen: dict[str, int] = {}
    for b in f.iter_batches(batch_size=300000,
                            columns=["inchikey14", "precursor_mz", "instrument_type"]):
        d = b.to_pandas()
        ok = ((d.instrument_type == "timsTOF")
              & d.precursor_mz.between(MZ_LO, MZ_HI)
              & d.inchikey14.notna())
        for k in d.inchikey14[ok]:
            seen[k] = seen.get(k, 0) + 1
    # The real test averages three spectra per molecule; ask for at least two.
    pool = [k for k, n in seen.items() if n >= 2]
    print(f"eligible molecules: {len(pool)}")
    held = set(RNG.choice(pool, size=min(N_MOL, len(pool)), replace=False))

    # Second pass: write the library without them, and a query set from them.
    lib_head = open(ROOT / "lib_head.bin", "wb")
    lib_peaks = open(ROOT / "lib_peaks.bin", "wb")
    qry_head = open(ROOT / "qry_head.bin", "wb")
    qry_peaks = open(ROOT / "qry_peaks.bin", "wb")
    lib_mols: dict[str, int] = {}
    lib_names: list[str] = []
    lib_smiles: list[str] = []
    qry_mols: dict[str, int] = {}
    qry_names: list[str] = []
    qry_truth: list[str] = []
    lib_off = qry_off = 0
    nl = nq = 0

    cols = ["inchikey14", "normalized_smiles", "precursor_mz", "adduct",
            "instrument_type", "ms2_mzs", "ms2_normalized_intensities"]
    for b in f.iter_batches(batch_size=100000, columns=cols):
        d = b.to_pandas()
        for r in d.itertuples(index=False):
            key = r.inchikey14 if isinstance(r.inchikey14, str) else ""
            if not key:
                continue
            mz = np.asarray(r.ms2_mzs, dtype=np.float32)
            it = np.asarray(r.ms2_normalized_intensities, dtype=np.float32)
            k = min(len(mz), len(it))
            if k == 0:
                continue
            o = np.argsort(mz[:k])
            mz, it = mz[:k][o], it[:k][o]
            tag = (ADDUCT_ID.get(r.adduct, -1) * 16
                   + INSTR_ID.get(r.instrument_type, -1))
            pmz = float(r.precursor_mz) if r.precursor_mz == r.precursor_mz else 0.0
            smi = r.normalized_smiles if isinstance(r.normalized_smiles, str) else ""

            # A held-out molecule contributes its first eligible spectrum to
            # the queries; everything else it has stays in the library.
            take = False
            if key in held and r.instrument_type == "timsTOF" and MZ_LO <= pmz <= MZ_HI:
                take = key not in qry_mols
            if take:
                if key not in qry_mols:
                    qry_mols[key] = len(qry_names)
                    qry_names.append(key)
                    qry_truth.append(smi)
                qry_peaks.write(mz.tobytes()); qry_peaks.write(it.tobytes())
                qry_head.write(struct.pack("<fiiii", pmz, qry_mols[key],
                                           qry_off, k, tag))
                qry_off += k
                nq += 1
            else:
                if key not in lib_mols:
                    lib_mols[key] = len(lib_names)
                    lib_names.append(key)
                    lib_smiles.append(smi)
                elif smi and not lib_smiles[lib_mols[key]]:
                    lib_smiles[lib_mols[key]] = smi
                lib_peaks.write(mz.tobytes()); lib_peaks.write(it.tobytes())
                lib_head.write(struct.pack("<fiiii", pmz, lib_mols[key],
                                           lib_off, k, tag))
                lib_off += k
                nl += 1
        print(f"  library {nl}  queries {nq}", end="\r", flush=True)

    for h in (lib_head, lib_peaks, qry_head, qry_peaks):
        h.close()
    (ROOT / "lib_mols.txt").write_text(
        "\n".join(f"{a}\t{b}" for a, b in zip(lib_names, lib_smiles)), encoding="utf-8")
    (ROOT / "qry_mols.txt").write_text(
        "\n".join(f"{a}\t{b}" for a, b in zip(qry_names, qry_truth)), encoding="utf-8")
    print(f"\nlibrary: {nl} spectra over {len(lib_names)} molecules")
    print(f"queries: {nq} spectra over {len(qry_names)} molecules")


main()
