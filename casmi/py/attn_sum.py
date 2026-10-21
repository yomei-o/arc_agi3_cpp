"""Summarise attn_eval logs (casmi/cpp/attn): per log, the model baselines, best attn config per vocab,
and for each evaluated config the attn-alone / shuffled / ensemble rows (MRR_in, MRR_all).
usage: python attn_sum.py log1 [log2 ...]"""
import re
import sys

ROW = re.compile(r"^\s+(.*?)\s+MRR_in ([0-9.]+)\s+MRR_all ([0-9.]+)")


def summarise(path):
    print(f"== {path}")
    cfg = None
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.rstrip()
        if line.startswith(" model ") or line.startswith(" models"):
            cfg = None
        if line.startswith(" best attn") or "notwin" in line or "diag" in line and "q," in line:
            print(line)
            continue
        if line.startswith(" config"):
            cfg = line.strip()
            print(" " + cfg)
            continue
        m = ROW.match(line)
        if not m:
            continue
        lab, mi, ma = m.group(1), m.group(2), m.group(3)
        if lab.startswith("attn ") and " k=" in lab:
            continue  # grid rows (attn <vocab> <sim> k= T= lam=)
        print(f"   {lab:44s} {mi} / {ma}")


for p in sys.argv[1:]:
    summarise(p)
