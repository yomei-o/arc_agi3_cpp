r"""q_extract.py <root> <qry> [<qry> ...]

Holdout query spectra (<qry>_head/_peaks/_mols.bin, written by make_holdout.py) carry no collision energy or
ionization mode, which ICEBERG / GLACIER need as inputs. Recover them from train.parquet: every query spectrum
is a train row of its molecule, matched by float32 precursor, adduct / instrument tag and the floor-filtered
sorted peak list (export_bin.py's processing). Writes <root>\iceberg\q_<qry>.json:
  {mol: [{prec, adduct, instr, mode, ce: [..] | null, mz: [..], it: [..]}, ...]}  (peaks = the bin's peaks)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pyarrow.compute as pc

ROOT = Path(sys.argv[1])
ADDUCTS = ["[M+H]+", "[M-H]-", "[M+Na]+", "[M+NH4]+", "[M+K]+", "[M+Cl]-",
           "[M+CH2O2-H]-", "[M+H-H2O]+", "[2M+H]+", "[2M+Na]+"]
INSTR = ["timsTOF", "Orbitrap", "QTOF", "LC-ESI-QTOF", "ESI-QFT"]
HEAD = np.dtype([("precursor", "<f4"), ("mol", "<i4"), ("off", "<i4"), ("n", "<i4"), ("tag", "<i4")])
FLOOR = 0.002


def proc(mz, it):
    mz = np.asarray(mz, np.float32)
    it = np.asarray(it, np.float32)
    k = min(len(mz), len(it))
    mz, it = mz[:k], it[:k]
    if k and it.max() > 0:
        keep = it >= FLOOR * it.max()
        mz, it = mz[keep], it[keep]
    o = np.argsort(mz)
    return mz[o], it[o]


for qry in sys.argv[2:]:
    h = np.fromfile(ROOT / f"{qry}_head.bin", dtype=HEAD)
    peaks = np.fromfile(ROOT / f"{qry}_peaks.bin", dtype="<f4")
    mols = [l.split("\t")[0] for l in (ROOT / f"{qry}_mols.txt").read_text(encoding="utf-8").splitlines()]
    want = sorted(set(mols[m] for m in h["mol"]))
    t = pq.read_table(ROOT / "train.parquet",
                      columns=["inchikey14", "precursor_mz", "adduct", "instrument_type", "ionization_mode",
                               "collision_energy_ev", "ms2_mzs", "ms2_normalized_intensities"],
                      filters=[("inchikey14", "in", want)]).to_pandas()
    by = {}
    for r in t.itertuples(index=False):
        by.setdefault(r.inchikey14, []).append(r)
    out, nmatch, nce = {}, 0, 0
    for q in h:
        name = mols[q["mol"]]
        k = int(q["n"])
        o = int(q["off"])
        # export_bin layout: per spectrum k m/z then k intensities, `off` counts peaks -> float offset 2*off
        mz = peaks[2 * o:2 * o + k]
        it = peaks[2 * o + k:2 * o + 2 * k]
        ad, ins = int(q["tag"]) // 16, int(q["tag"]) % 16
        if ins == 15:  # instrument id -1 (unknown): tag = 16*adduct - 1
            ad, ins = ad + 1, -1
        hit = None
        for r in by.get(name, []):
            if np.float32(r.precursor_mz) != q["precursor"]:
                continue
            if r.adduct != (ADDUCTS[ad] if ad >= 0 else r.adduct):
                continue
            pm, pi = proc(r.ms2_mzs, r.ms2_normalized_intensities)
            if len(pm) == k and np.array_equal(pm, mz):
                hit = r
                break
        ce = None
        mode = None
        adduct = ADDUCTS[ad] if 0 <= ad < len(ADDUCTS) else None
        instr = INSTR[ins] if 0 <= ins < len(INSTR) else None
        if hit is not None:
            nmatch += 1
            mode = hit.ionization_mode
            adduct = hit.adduct
            instr = hit.instrument_type if isinstance(hit.instrument_type, str) else instr
            if hit.collision_energy_ev is not None and len(hit.collision_energy_ev):
                ce = [float(x) for x in hit.collision_energy_ev]
                nce += 1
        out.setdefault(name, []).append(dict(prec=float(q["precursor"]), adduct=adduct, instr=instr, mode=mode,
                                             ce=ce, mz=mz.tolist(), it=it.tolist()))
    (ROOT / "iceberg" / f"q_{qry}.json").write_text(json.dumps(out))
    print(f"{qry}: {len(h)} spectra, {len(out)} molecules, matched {nmatch}, with CE {nce}", flush=True)
