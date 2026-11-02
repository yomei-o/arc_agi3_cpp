"""Benchmark I (phase by 1/sqrt(x)) - own design v1: table of quadratic segments, no Newton step.

M = x_norm / 2^20 in [1, 4) after shifting out s leading zero pairs (1/sqrt(x) = 2^(s - n_p) / sqrt(M)).
The top A bits of x_norm address a segment; t = the next D bits (a fraction of the segment, [0, 1)).
y = c0 - T(r t),  r = c1 - T(c2 t)   (c1, c2 magnitudes; 1/sqrt is decreasing and convex)
y has WY fractional bits and approximates 1/sqrt(M) in (1/2, 1]; the phase register is Y 2^s with
F = WY + n_p fractional bits of a turn.

T(.) truncates every partial product below the target's lsb (as the circuit does with _add_row).
This model is the exact integer arithmetic of the circuit; `check` runs it over every x.
"""
from __future__ import annotations
import dataclasses, math, sys
import numpy as np

N_P = 10
WIDTH = 2 * N_P + 2
MAX_X = 3 * (2**N_P - 2)**2


@dataclasses.dataclass(frozen=True)
class P:
    A: int = 9    # address bits (top bits of x_norm)
    D: int = 13   # bits of t used
    WY: int = 27  # fractional bits of y and c0
    W1: int = 22  # fractional bits of c1 / r (in units of y per unit t)
    W2: int = 14  # fractional bits of c2


def fit_table(p: P):
    """Per-address minimax-ish (Chebyshev-node) quadratic in t for 1/sqrt(M) on the segment."""
    rows = []
    seg_w = 4.0 / 2**p.A
    for a in range(2**p.A):
        lo = a * seg_w
        if lo < 1.0:
            rows.append((0, 0, 0))
            continue
        nodes = np.cos((2 * np.arange(40) + 1) * np.pi / 80)
        t = (nodes + 1) / 2
        f = 1 / np.sqrt(lo + seg_w * t)
        # relative-error weighted least squares on Chebyshev nodes ~ minimax
        V = np.stack([np.ones_like(t), -t, t * t], 1) * np.sqrt(lo + seg_w * t)[:, None]
        c, *_ = np.linalg.lstsq(V, np.ones_like(t), rcond=None)
        rows.append((round(c[0] * 2**p.WY), round(c[1] * 2**p.W1), round(c[2] * 2**p.W2)))
    return rows


def model(p: P, xs: np.ndarray, table=None) -> np.ndarray:
    """Integer Y (units 2^-WY) times 2^s, i.e. the phase register, for each x."""
    table = fit_table(p) if table is None else table
    tab = np.array(table, dtype=np.int64)
    xs = xs.astype(np.int64)
    s = (WIDTH - (np.floor(np.log2(xs)).astype(np.int64) + 1)) // 2
    xn = xs << (2 * s)
    a = xn >> (WIDTH - p.A)
    rest = WIDTH - p.A
    tb = (xn >> (rest - p.D)) & ((1 << p.D) - 1) if p.D <= rest else (xn & ((1 << rest) - 1)) << (p.D - rest)
    c0, c1, c2 = tab[a, 0], tab[a, 1], tab[a, 2]
    # T(c2 t) in units of 2^-W1: sum over t bits (weight 2^-m, m = 1..D) of c2 * 2^(W1 - W2 - m), each row truncated
    def trunc_prod(coef, wcoef, wout):
        acc = np.zeros_like(xs)
        for m in range(1, p.D + 1):
            bitm = (tb >> (p.D - m)) & 1
            e = wout - wcoef - m
            acc += bitm * (coef << e if e >= 0 else coef >> -e)
        return acc
    r = c1 - trunc_prod(c2, p.W2, p.W1)
    y = c0 - trunc_prod(r, p.W1, p.WY)
    return y << s, s


def check(p: P, verbose=True):
    xs = np.arange(1, MAX_X + 1, dtype=np.int64)
    Y, s = model(p, xs)
    approx = Y / 2.0**(p.WY + N_P)
    rel = np.abs(approx * np.sqrt(xs.astype(np.float64)) - 1)
    worst = rel.max()
    if verbose:
        print(f"{p}: max rel err 2^{math.log2(worst):.2f} at x={rel.argmax() + 1} (need < 2^-24, phase rounding extra)")
    return worst


if __name__ == "__main__":
    kw = {k: int(v) for k, v in (a.split("=") for a in sys.argv[1:])}
    check(P(**kw))
