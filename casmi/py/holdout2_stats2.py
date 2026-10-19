"""Summaries of h2_molstats.tsv (written by holdout2_stats.py)."""
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else r"C:\prog\casmi")
rows = []
for l in open(ROOT / "h2_molstats.tsv", encoding="utf-8"):
    if l.startswith("key\t"):
        continue
    k, na, nt, nta, coc, old, libs = l.rstrip("\n").split("\t")
    lib = dict((x.rsplit(":", 1)[0], int(x.rsplit(":", 1)[1])) for x in libs.split(",") if x)
    rows.append((k, int(na), int(nt), int(nta), int(coc), old, lib))

mol_by_lib = Counter(l for r in rows for l in r[6])
spec_by_lib = Counter()
for r in rows:
    spec_by_lib.update(r[6])
print("molecules per ingest lib:", mol_by_lib.most_common())
print("spectra per ingest lib:", spec_by_lib.most_common())
print("old holdout membership:", Counter(r[5] for r in rows if r[5]))
for name, sel in [("coconut", lambda r: r[4] == 1),
                  ("enveda-np-examples", lambda r: "enveda-np-examples" in r[6]),
                  ("enveda-180", lambda r: "enveda-180" in r[6]),
                  ("timsTOF-range>=1", lambda r: r[2] >= 1)]:
    S = [r for r in rows if sel(r)]
    print(f"{name}: {len(S)} mols, coconut {sum(r[4] for r in S)}, tims-range>=1 {sum(r[2] >= 1 for r in S)},"
          f" n_all mean {sum(r[1] for r in S)/len(S):.2f}, only-one-lib {sum(len(r[6]) == 1 for r in S)},"
          f" libs {Counter(l for r in S for l in r[6]).most_common(6)}")
