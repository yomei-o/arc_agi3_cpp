"""e2e: print the schema of the real test/train parquet and the adduct/instrument vocab of test."""
import sys
import pyarrow.parquet as pq

root = sys.argv[1] if len(sys.argv) > 1 else r"C:\prog\casmi"
for name in ("test.parquet", "train.parquet"):
    f = pq.ParquetFile(root + "\\" + name)
    print("==", name, f.metadata.num_rows, "rows,", f.metadata.num_row_groups, "row groups")
    print(f.schema_arrow)
t = pq.read_table(root + "\\test.parquet").to_pandas()
print(t.head(3).T)
for c in t.columns:
    if t[c].dtype == object and not hasattr(t[c].iloc[0], "__len__") or c in ("adduct", "instrument_type", "polarity", "collision_energy"):
        try:
            print(c, t[c].value_counts(dropna=False).head(20).to_dict())
        except Exception as e:
            print(c, "?", e)
print("molecules", t.molecule_id.nunique(), "spectra", len(t))
print("precursor decimals check:", (t.precursor_mz.round(4) == t.precursor_mz).mean())
