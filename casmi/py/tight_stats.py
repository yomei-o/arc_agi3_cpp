import csv, sys
rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
n = [int(r["tight_count"]) for r in rows]
import collections
c = collections.Counter(n)
print("total molecules:", len(n))
print("zero tight candidates:", sum(1 for x in n if x == 0),
      f"({100.0*sum(1 for x in n if x==0)/len(n):.1f}%)")
for thresh in (1, 3, 5, 10, 25):
    print(f"  tight_count < {thresh}: {sum(1 for x in n if x < thresh)}")
