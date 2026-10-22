"""fte_sum.py <log> [<log> ...] : compact table of tanimoto MRR_in per model / ensemble from cnn_win_eval2 logs
(ft_eval.bat output: fte_<stem>_<qry>.log). Columns = logs, rows = models in the order of the first log."""
import re
import sys


def parse(path):
    rows, cur = {}, None
    order = []
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re.match(r"^ model (.+)$", line.rstrip("\n"))
        if m:
            cur = m.group(1).strip()
            cur = re.sub(r"pretrain\\\S+?_(hcqry|hxqry|haqry)_mol\.bin", "FT", cur)
            order.append(cur)
            continue
        m = re.match(r"^\s+tanimoto\s+MRR_in (\S+)", line)
        if m and cur:
            rows[cur] = float(m.group(1))
    return order, rows


def main():
    logs = sys.argv[1:]
    parsed = [parse(p) for p in logs]
    order = parsed[0][0]
    names = [re.sub(r".*fte_", "", p).replace(".log", "") for p in logs]
    print("%-46s" % "tanimoto MRR_in" + "".join("%22s" % n[-21:] for n in names))
    for r in order:
        print("%-46s" % r[:46] + "".join("%22s" % ("%.4f" % d[r] if r in d else "-") for _, d in parsed))


if __name__ == "__main__":
    main()
