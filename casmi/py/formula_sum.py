"""Summarise formula_eval logs (cpp/formula/formula_eval.cpp).

python formula_sum.py <log> [N] [--best k]
Prints, for re-rank depth N (default: the largest in the log), every feature's best weight (with its shuffled
control at the same weight), and the "base + a ll + b f" grid.
"""
import re
import sys


def parse(path):
    data = {}
    n = None
    sec = ""
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re.match(r"=== (\S+), re-rank of the top (\d+)", line)
        if m:
            n = int(m.group(2))
            data[n] = []
            sec = ""
            continue
        m = re.match(r" \[(.*)\]\s*$", line)
        if m and n is not None:
            sec = m.group(1)
            continue
        m = re.match(r"  (.*?)\s+MRR_in ([\d.]+)\s+MRR_all ([\d.]+)(?:\s+d ([+-][\d.]+) \(SE ([\d.]+)\) up (\d+) down (\d+))?", line)
        if m and n is not None:
            lab = m.group(1).strip()
            d = float(m.group(4)) if m.group(4) else 0.0
            se = float(m.group(5)) if m.group(5) else 0.0
            data[n].append((sec, lab, float(m.group(2)), float(m.group(3)), d, se))
    return data


def main():
    path = sys.argv[1]
    data = parse(path)
    n = int(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("-") else max(data)
    rows = data[n]
    base = [r for r in rows if r[1].startswith("base: ")][0]
    print(f"{path}  N={n}  base MRR_in {base[2]:.4f} MRR_all {base[3]:.4f}")
    secs = []
    for r in rows:
        if r[0] and r[0] not in secs:
            secs.append(r[0])
    print(f"{'feature':44s} {'alone':>7s} {'SHUF':>7s} {'best w':>7s} {'MRR_in':>7s} {'d':>8s} {'SE':>6s} {'SHUF d':>8s}")
    for s in secs:
        rs = [r for r in rows if r[0] == s]
        alone = [r for r in rs if r[1] == "alone"]
        sh = [r for r in rs if r[1] == "SHUF alone"]
        comb = [r for r in rs if re.match(r"base \+ [\d.]+ z$", r[1])]
        if not comb:
            continue
        b = max(comb, key=lambda r: r[2])
        w = b[1].split()[2]
        shc = [r for r in rs if r[1] == f"SHUF base + {w} z"]
        print(f"{s:44s} {alone[0][2]:7.4f} {sh[0][2]:7.4f} {w:>7s} {b[2]:7.4f} {b[4]:+8.4f} {b[5]:6.4f} {shc[0][4] if shc else 0:+8.4f}")
    grid = [r for r in rows if r[0].startswith("base + a z(phys_ll)")]
    if grid:
        print("\nbase + a z(phys_ll) + b z(f) grid (MRR_in, d vs base):")
        for r in grid:
            print(f"  {r[1]:52s} {r[2]:.4f} {r[3]:.4f} {r[4]:+.4f} ({r[5]:.4f})")


if __name__ == "__main__":
    main()
