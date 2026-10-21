"""e2e: fake competition input from the holdouts hcqry (denovo) + hxqry (xinstr).

    python e2e_make.py <casmi dir> <out comp dir>

Writes <out>/train.parquet (= zlib: train minus every spectrum of the ha/hb/hc denovo molecules, minus the
has/hbs sibling query spectra, minus the timsTOF spectra of the hx molecules), <out>/test.parquet (the
hcqry + hxqry query spectra, raw peaks as in train, real-test column set and dtypes, molecule_id m_xxxxxx),
<out>/sample_submission.csv and <out>/truth.tsv (molecule_id, set, inchikey14, truth smiles).

Query spectra are identified exactly as build_ylib.py does: (inchikey14, tag, n peaks after the 0.002 floor,
float32 precursor bytes, float32 peak bytes), with make_holdout's cleaning (argsort m/z, then floor).
"""
import struct, sys, random
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

D = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
FLOOR = 0.002
ADDUCTS = ["[M+H]+", "[M-H]-", "[M+Na]+", "[M+NH4]+", "[M+K]+", "[M+Cl]-",
           "[M+CH2O2-H]-", "[M+H-H2O]+", "[2M+H]+", "[2M+Na]+"]
ADDUCT_ID = {a: i for i, a in enumerate(ADDUCTS)}
INSTR = ["timsTOF", "Orbitrap", "QTOF", "LC-ESI-QTOF", "ESI-QFT"]
INSTR_ID = {a: i for i, a in enumerate(INSTR)}


def names(stem):
    return [l.rstrip("\r\n").split("\t") for l in open(D / f"{stem}_mols.txt", encoding="utf-8")]


def qkeys(stem):
    nm = names(stem)
    hb = (D / f"{stem}_head.bin").read_bytes()
    pk = (D / f"{stem}_peaks.bin").read_bytes()
    c = Counter()
    for i in range(len(hb) // 20):
        p, mol, off, n, tag = struct.unpack_from("<fiiii", hb, 20 * i)
        c[(nm[mol][0], tag, n, struct.pack("<f", p), pk[8 * off:8 * (off + n)])] += 1
    return c, nm


want = {}          # set name -> Counter of query keys (rows go to test.parquet)
truth = {}
for s in ("hcqry", "hxqry"):
    want[s], nm = qkeys(s)
    truth[s] = nm
drop_spec = Counter()  # sibling query spectra: dropped from train
for s in ("hasqry", "hbsqry"):
    c, _ = qkeys(s)
    drop_spec.update(c)
drop_mol = set()
for s in ("haqry", "hbqry", "hcqry"):
    drop_mol |= {r[0] for r in names(s)}
hx_mol = {r[0] for r in truth["hxqry"]}
check_mol = {k[0] for c in list(want.values()) + [drop_spec] for k in c}
print("query spectra wanted:", {s: sum(c.values()) for s, c in want.items()}, "sibling drops", sum(drop_spec.values()),
      "denovo molecules", len(drop_mol), flush=True)


def key_of(r):
    mz = np.asarray(r["ms2_mzs"], dtype=np.float32)
    it = np.asarray(r["ms2_normalized_intensities"], dtype=np.float32)
    k = min(len(mz), len(it))
    if k == 0:
        return None
    o = np.argsort(mz[:k])
    mz, it = mz[:k][o], it[:k][o]
    if len(it) and FLOOR > 0:
        keep = it >= FLOOR * it.max()
        mz, it = mz[keep], it[keep]
    if len(mz) == 0:
        return None
    tag = ADDUCT_ID.get(r["adduct"], -1) * 16 + INSTR_ID.get(r["instrument_type"], -1)
    pmz = float(r["precursor_mz"]) if r["precursor_mz"] == r["precursor_mz"] else 0.0
    return (r["inchikey14"], tag, len(mz), struct.pack("<f", pmz), mz.tobytes() + it.tobytes())


f = pq.ParquetFile(D / "train.parquet")
writer = None
test_rows = {s: [] for s in want}
n_in = n_out = n_dm = n_ds = n_hx = 0
for b in f.iter_batches(batch_size=200000):
    t = pa.Table.from_batches([b])
    ik = t.column("inchikey14").to_pylist()
    ins = t.column("instrument_type").to_pylist()
    keep = np.ones(len(ik), dtype=bool)
    idx_check = [i for i, k in enumerate(ik) if k in check_mol]
    rows = t.take(pa.array(idx_check, type=pa.int64())).to_pylist() if idx_check else []
    for i, r in zip(idx_check, rows):
        k = key_of(r)
        if k is None:
            continue
        hit = False
        for s, c in want.items():
            if c.get(k, 0) > 0:
                c[k] -= 1; test_rows[s].append(r); hit = True; break
        if not hit and drop_spec.get(k, 0) > 0:
            drop_spec[k] -= 1; n_ds += 1; keep[i] = False
        if hit:
            keep[i] = False
    for i, k in enumerate(ik):
        if not keep[i]:
            continue
        if k in drop_mol:
            keep[i] = False; n_dm += 1
        elif k in hx_mol and ins[i] == "timsTOF":
            keep[i] = False; n_hx += 1
    n_in += len(ik)
    kt = t.filter(pa.array(keep))
    n_out += kt.num_rows
    if writer is None:
        writer = pq.ParquetWriter(OUT / "train.parquet", t.schema)
    writer.write_table(kt)
    print(f"  {n_in} rows in, {n_out} out", flush=True)
writer.close()
print("unmatched query spectra:", {s: sum(c.values()) for s, c in want.items()}, "unmatched sibling",
      sum(drop_spec.values()), "dropped denovo", n_dm, "sibling", n_ds, "hx timsTOF", n_hx, flush=True)

# test.parquet: real-test schema
real = pq.read_schema(D / "test.parquet")
rnd = random.Random(2026)
ids = rnd.sample(range(16 ** 6), 2000)
mid_of, tr = {}, []
j = 0
for s in ("hcqry", "hxqry"):
    for name, smi in truth[s]:
        mid_of[(s, name)] = "m_%06x" % ids[j]; j += 1
        tr.append((mid_of[(s, name)], s, name, smi))
recs = []
sid = rnd.sample(range(16 ** 8), 5000)
for s in ("hcqry", "hxqry"):
    for r in test_rows[s]:
        recs.append({
            "molecule_id": mid_of[(s, r["inchikey14"])],
            "spectrum_id": "s_%08x" % sid[len(recs)],
            "ms2_mzs": r["ms2_mzs"], "ms2_normalized_intensities": r["ms2_normalized_intensities"],
            "base_peak_intensity": r["base_peak_intensity"], "adduct": r["adduct"],
            "ionization_mode": r["ionization_mode"], "instrument_type": r["instrument_type"],
            "precursor_mz": r["precursor_mz"], "collision_energy_orig": r["collision_energy_orig"],
            "collision_energy_ev": r["collision_energy_ev"],
            "collision_energy_orig_units": r["collision_energy_orig_units"]})
recs.sort(key=lambda x: x["molecule_id"])
tt = pa.Table.from_pylist(recs, schema=pa.schema([real.field(n) for n in real.names]))
pq.write_table(tt, OUT / "test.parquet")
mols = sorted({r["molecule_id"] for r in recs})
(OUT / "sample_submission.csv").write_text("molecule_id,smiles\n" + "".join(f"{m},CCO\n" for m in mols))
(OUT / "truth.tsv").write_text("".join(f"{a}\t{b}\t{c}\t{d}\n" for a, b, c, d in tr), encoding="utf-8")
print("test:", len(recs), "spectra,", len(mols), "molecules; truth rows", len(tr))
print(pq.read_schema(OUT / "test.parquet"))
