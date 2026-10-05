import sys
p = sys.argv[1]
for line in open(p, encoding="utf-8").read().splitlines()[:8]:
    a, _, b = line.partition("\t")
    print(repr(a), len(a), b[:50])
