"""fw_domain.py: does the public ICEBERG / GLACIER (trained on MassSpecGym Orbitrap/QTOF) lose discrimination on
timsTOF? For each instrument, sample [M+H]+ spectra of molecules with >= 9 same-formula isomers in train.parquet,
candidates = the true molecule + 9 random isomers, score with fw_score and report the MRR of the truth per model.

    python fw_domain.py make <train.parquet> <fw_in.txt> [n per instrument=400]
    python fw_domain.py eval <fw_in.txt> <fw_out.tsv>
"""
import random, sys
from collections import defaultdict

INSTR = ("timsTOF", "Orbitrap", "QTOF")


def make(pq, out, n):
    import pandas as pd
    d = pd.read_parquet(pq, columns=["ingest_lib", "normalized_smiles", "inchikey14", "molecular_formula", "ionization_mode",
                                     "instrument_type", "adduct", "precursor_mz", "ms2_mzs", "ms2_normalized_intensities",
                                     "collision_energy_ev", "collision_energy_orig", "collision_energy_orig_units"])
    iso = defaultdict(dict)
    for f, k, s in zip(d.molecular_formula, d.inchikey14, d.normalized_smiles):
        iso[f].setdefault(k, s)
    rng = random.Random(7)
    with open(out, "w", encoding="utf-8") as fo:
        for ins in INSTR:
            sub = d[(d.instrument_type == ins) & (d.adduct == "[M+H]+")]
            idx = [i for i, f in zip(sub.index, sub.molecular_formula) if len(iso[f]) >= 10]
            rng.shuffle(idx)
            seen = set()
            k = 0
            for i in idx:
                r = d.loc[i]
                if r.inchikey14 in seen:
                    continue
                seen.add(r.inchikey14)
                others = [s for kk, s in iso[r.molecular_formula].items() if kk != r.inchikey14]
                cands = [r.normalized_smiles] + rng.sample(others, 9)
                fo.write(f"M {ins}|{r.ingest_lib}|{k}\n")
                for s in cands:
                    fo.write(f"C {s}\n")
                ce = ",".join(repr(float(x)) for x in r.collision_energy_ev) if r.collision_energy_ev is not None else ""
                pk = ",".join(f"{float(a)!r}:{float(b)!r}" for a, b in zip(r.ms2_mzs, r.ms2_normalized_intensities))
                f_ = lambda v: "" if v is None or v != v else str(v)
                fo.write("S " + "\t".join([r.adduct, f_(r.ionization_mode), ins, f_(r.collision_energy_orig_units),
                                          f_(r.collision_energy_orig), ce, repr(float(r.precursor_mz)), pk]) + "\n")
                k += 1
                if k >= n:
                    break
            print(ins, k, flush=True)


def ev(fin, fout):
    truth = {}
    mid = None
    for l in open(fin, encoding="utf-8"):
        if l.startswith("M "):
            mid = l[2:].strip()
        elif l.startswith("C ") and mid not in truth:
            truth[mid] = l[2:].rstrip("\r\n").split("\t")[0]
    sc = defaultdict(dict)
    for l in open(fout, encoding="utf-8"):
        p = l.rstrip("\n").split("\t")
        if len(p) >= 4:
            sc[p[0]][p[1]] = (float(p[2]) if p[2] != "nan" else None, float(p[3]) if p[3] != "nan" else None)
    agg = defaultdict(lambda: defaultdict(list))
    for m, t in truth.items():
        ins, lib = m.split("|")[:2]
        for j, nm in ((0, "gl"), (1, "ice")):
            v = {s: x[j] for s, x in sc[m].items() if x[j] is not None}
            if t not in v or len(v) < 5:
                continue
            rank = 1 + sum(1 for s, x in v.items() if s != t and x > v[t])
            for key in (ins, ins + "|" + lib):
                agg[key][nm].append(1.0 / rank)
    print("MRR of the truth among itself + 9 same-formula isomers (random = 0.293)")
    for key in sorted(agg, key=lambda k: (k.count("|"), k)):
        a = agg[key]
        if len(a["ice"]) < 30 and "|" in key:
            continue
        print(f"{key:40s} n {len(a['ice']):4d}  ice {sum(a['ice']) / max(1, len(a['ice'])):.3f}  "
              f"gl {sum(a['gl']) / max(1, len(a['gl'])):.3f}")


if __name__ == "__main__":
    if sys.argv[1] == "make":
        make(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 400)
    else:
        ev(sys.argv[2], sys.argv[3])
