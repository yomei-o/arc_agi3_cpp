r"""Combine the direct spectral search with the analog-propagation backfill.

match.cpp's staged widening fills exactly 25 slots almost regardless of
whether a real match exists nearby, by widening the mass window until it
does - so "already has 25 candidates" cannot tell a confident Class-1 hit
apart from 25 slots of padding found only because the window had to grow
huge. tight_count.csv is the separate signal: how many candidates the
TIGHTEST pass alone found, before any widening. Those are kept; the rest of
each molecule's 25 slots are filled from the analog-propagation list instead
of direct's own widened guesses, since analog candidates are chosen by
chemical similarity rather than "whatever was left in a much bigger window",
and only direct's widened leftovers pad out anything analog still could not
fill.

    python merge_submissions.py <direct.csv> <analog.csv> <tight_count.csv> <out.csv> [topk]
"""
from __future__ import annotations

import csv, sys
from pathlib import Path

direct_path, analog_path, tight_path, out_path = map(Path, sys.argv[1:5])
K = int(sys.argv[5]) if len(sys.argv) > 5 else 25

direct = {r[0]: r[1] for r in csv.reader(direct_path.open(encoding="utf-8"))}
analog = {r[0]: r[1] for r in csv.reader(analog_path.open(encoding="utf-8"))}
tight = {r[0]: int(r[1]) for r in csv.reader(tight_path.open(encoding="utf-8")) if r[0] != "molecule_id"}
direct.pop("molecule_id", None)
analog.pop("molecule_id", None)

rows = []
n_from_analog = []
for mol_id, dsmi in direct.items():
    d_list = [s for s in dsmi.split(";") if s]
    keep_n = min(tight.get(mol_id, K), len(d_list), K)
    kept = d_list[:keep_n]
    seen = set(kept)
    a_list = [s for s in analog.get(mol_id, "").split(";") if s and s not in seen]
    merged = kept + a_list
    if len(merged) < K:
        extra = [s for s in d_list[keep_n:] if s not in seen and s not in merged]
        merged += extra
    merged = merged[:K]
    n_from_analog.append(min(len(a_list), K - keep_n))
    rows.append((mol_id, ";".join(merged) if merged else "CCO"))

with out_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["molecule_id", "smiles"])
    w.writerows(rows)
import numpy as np
n = np.array(n_from_analog)
print(f"wrote {len(rows)} rows; slots filled from analog propagation: "
      f"mean {n.mean():.1f}, molecules with >=1 analog slot: {(n>0).sum()}")
