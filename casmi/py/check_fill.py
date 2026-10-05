import csv, sys
rows = list(csv.reader(open(sys.argv[1] if len(sys.argv)>1 else r"C:\prog\casmi\submission.csv", encoding="utf-8")))[1:]
n = [len(r[1].split(";")) for r in rows]
print("rows", len(rows), "full25", sum(1 for x in n if x == 25), "min", min(n))
