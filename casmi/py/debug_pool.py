import sys
from collections import Counter
path = sys.argv[1]
lines = open(path, encoding="utf-8").read().splitlines()
print("total lines", len(lines))
smis = [l.split("\t", 1)[1] if "\t" in l else "" for l in lines]
print("unique smiles", len(set(smis)))
print("sample [0]:", lines[0][:80])
print("sample [275810]:", lines[275810][:80])
print("sample [500000]:", lines[500000][:80])
c = Counter(smis)
print("most common:", c.most_common(3))
