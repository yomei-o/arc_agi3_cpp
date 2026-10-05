r"""Carve a test set out of the training data, shaped like the real one.

The real test is 400 molecules, 1,213 spectra, every one of them timsTOF, with
precursor masses between 245 and 460. A holdout that does not look like that
would measure the wrong thing.

Two modes, and which one is right depends on a fact about the competition that
took a while to pin down:

  --mode sibling  the held-out molecule keeps its other spectra in the library,
                  so the exact answer is there to be found. This measures
                  retrieval: can the spectrum pick its own molecule out of the
                  hundred that share its mass.

  --mode denovo   every spectrum of the held-out molecule leaves the library,
                  so the exact answer is NOT there. This is the one that
                  matches the competition: Enveda generated the 400 test
                  molecules on their own platform and none of their spectra
                  have been published, and the official tutorial is a de novo
                  SMILES generator scored by Tanimoto similarity, not by exact
                  match. Under denovo a library search can only ever return a
                  neighbour, and the question is how close a neighbour.

The first holdout built here was denovo by accident, scored 0% exact match,
and was "fixed" into sibling. That was the wrong correction: the 0% was telling
the truth about the task.

    python make_holdout.py [root] [--mode sibling|denovo]
"""
from __future__ import annotations

import struct, sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

args = [a for a in sys.argv[1:] if not a.startswith("--")]
MODE = "sibling"
for a in sys.argv[1:]:
    if a.startswith("--mode"):
        MODE = a.split("=", 1)[1] if "=" in a else sys.argv[sys.argv.index(a) + 1]
if "--denovo" in sys.argv:
    MODE = "denovo"
assert MODE in ("sibling", "denovo"), MODE

ROOT = Path(args[0] if args else r"C:\prog\casmi")
PREFIX = "lib" if MODE == "sibling" else "dlib"
QPREFIX = "qry" if MODE == "sibling" else "dqry"
N_MOL = 400
MZ_LO, MZ_HI = 245.0, 461.0
RNG = np.random.default_rng(12345)

# Precursor jitter, in ppm, added to the query spectra only.
#
# Without this the holdout cannot see the biggest mistake it is capable of
# causing. train.parquet's precursor_mz is not an independent measurement:
# 63.5% of (molecule, adduct) groups carry bit-identical values and the 99th
# percentile spread is under 1 ppm, so it is computed from the formula and the
# adduct. A query carved out of train therefore matches its own library rows to
# the last digit, a 0.05 ppm window looks twenty times better than a 1 ppm one,
# and that window then deletes the answer on a real test set, where the
# precursor arrives from a different instrument. The field uses 10 ppm, and a
# published measurement of this competition's window recall reads 0.972 at
# 5 ppm and 1.000 at 10 ppm. Jittering the queries puts that error back.
JITTER_PPM = 0.0
for _a in sys.argv[1:]:
    if _a.startswith("--jitter"):
        JITTER_PPM = float(_a.split("=", 1)[1] if "=" in _a
                           else sys.argv[sys.argv.index(_a) + 1])

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
    print(f"mode {MODE}; jitter {JITTER_PPM} ppm; eligible molecules: {len(pool)}")
    held = set(RNG.choice(pool, size=min(N_MOL, len(pool)), replace=False))

    lib_head = open(ROOT / f"{PREFIX}_head.bin", "wb")
    lib_peaks = open(ROOT / f"{PREFIX}_peaks.bin", "wb")
    qry_head = open(ROOT / f"{QPREFIX}_head.bin", "wb")
    qry_peaks = open(ROOT / f"{QPREFIX}_peaks.bin", "wb")
    lib_mols: dict[str, int] = {}
    lib_names: list[str] = []
    lib_smiles: list[str] = []
    qry_mols: dict[str, int] = {}
    qry_names: list[str] = []
    qry_truth: list[str] = []
    lib_off = qry_off = 0
    nl = nq = 0
    MAX_Q = 3          # the real test averages three spectra per molecule
    # In sibling mode at least one eligible spectrum of a held molecule must
    # stay behind in the library, or the "sibling" is a sibling of nothing and
    # the holdout silently turns into denovo for every molecule that happens
    # to have exactly 2 or 3 eligible spectra. First cut of this script missed
    # that: MRR on a realistic 3-spectra-per-molecule holdout read 0.28
    # against single-spectrum's 0.47, which looked like the multi-spectrum
    # case being much harder when it was actually ~40% of molecules having
    # zero library representation.
    max_q_for: dict[str, int] = {}
    qcount: dict[str, int] = {}

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

            eligible = (r.instrument_type == "timsTOF" and MZ_LO <= pmz <= MZ_HI)
            if key in held:
                if key not in max_q_for:
                    cap = MAX_Q if MODE == "denovo" else min(MAX_Q, seen.get(key, 1) - 1)
                    max_q_for[key] = max(0, cap)
                # A query spectrum has to look like a test spectrum.
                if eligible and qcount.get(key, 0) < max_q_for[key]:
                    qcount[key] = qcount.get(key, 0) + 1
                    if key not in qry_mols:
                        qry_mols[key] = len(qry_names)
                        qry_names.append(key)
                        qry_truth.append(smi)
                    elif smi and not qry_truth[qry_mols[key]]:
                        qry_truth[qry_mols[key]] = smi
                    qry_peaks.write(mz.tobytes()); qry_peaks.write(it.tobytes())
                    qpmz = pmz * (1.0 + RNG.normal(0.0, JITTER_PPM * 1e-6))                         if JITTER_PPM else pmz
                    qry_head.write(struct.pack("<fiiii", qpmz, qry_mols[key],
                                               qry_off, k, tag))
                    qry_off += k
                    nq += 1
                    continue
                # In denovo the molecule is gone from the library entirely, so
                # its leftover spectra are dropped rather than filed away.
                if MODE == "denovo":
                    continue
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
    (ROOT / f"{PREFIX}_mols.txt").write_text(
        "\n".join(f"{a}\t{b}" for a, b in zip(lib_names, lib_smiles)), encoding="utf-8")
    (ROOT / f"{QPREFIX}_mols.txt").write_text(
        "\n".join(f"{a}\t{b}" for a, b in zip(qry_names, qry_truth)), encoding="utf-8")
    print(f"\nlibrary: {nl} spectra over {len(lib_names)} molecules -> {PREFIX}_*")
    print(f"queries: {nq} spectra over {len(qry_names)} molecules -> {QPREFIX}_*")


main()
