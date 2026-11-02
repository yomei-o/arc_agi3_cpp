"""Benchmark I circuit, own design v1 (see inv_model.py): normalize, table lookup of quadratic coefficients,
two Horner products, denormalize, per-bit phase, uncompute.

    python inv_circuit.py out.kmb [A=9 D=13 WY=30 W1=30 W2=28] [--check N]
--check N simulates N random x (plus edge cases) and compares the phase register with inv_model.
"""
from __future__ import annotations
import math, sys
from fractions import Fraction
import numpy as np
import kickmix as km
from nr_reference_circuit import _add_row, _normalize, _denormalize
from inv_model import P, N_P, WIDTH, MAX_X, fit_table, model


def widths(p: P, table):
    l0 = p.WY + 1 + N_P
    l1 = max(int(r[1]).bit_length() for r in table)
    l2 = max(int(r[2]).bit_length() for r in table)
    return l0, l1, l2


def _horner(b, inverse, p, x, c1, c2, y, scratch):
    rest = WIDTH - p.A
    tbits = [(x[rest - m], m) for m in range(1, p.D + 1)]

    def prod(acc, wacc, mult, wmult):
        for ctl, m in tbits:
            e = wacc - wmult - m
            terms = [(mult[i], i + e) for i in range(len(mult)) if 0 <= i + e < len(acc)]
            if terms:
                _add_row(b, inverse, ctl, terms, acc, scratch, subtract=True)
    steps = [(c1, p.W1, c2, p.W2), (y, p.WY, c1, p.W1)]
    for acc, wacc, mult, wmult in (reversed(steps) if inverse else steps):
        prod(acc, wacc, mult, wmult)


def build(p: P, compute_only=False):
    table = fit_table(p)
    l0, l1, l2 = widths(p, table)
    packed = [int(c0) | (int(c1) << l0) | (int(c2) << (l0 + l1)) for c0, c1, c2 in table]
    b = km.CircuitBuilder()
    x = b.create_quantum_register(WIDTH, name="x")
    alloc = (lambda n, nm: b.create_quantum_register(n, name=nm)) if compute_only else (lambda n, nm: b.alloc_qubits(n))
    flags = alloc(N_P, "flags")
    y = alloc(l0, "y1")
    c1 = alloc(l1, "c1")
    c2 = alloc(l2, "c2")
    scratch = alloc(l0, "scratch")
    addr = x[WIDTH - p.A:]
    target = km.array(list(y) + list(c1) + list(c2))

    def compute(inverse):
        if not inverse:
            with b.mark("normalize"):
                _normalize(b, False, x, flags)
            with b.mark("lookup"):
                b.init_lookup(table=packed, address=addr, target=target)
            with b.mark("horner"):
                _horner(b, False, p, x, c1, c2, y[:p.WY + 1], scratch)
            with b.mark("denormalize"):
                _denormalize(b, False, flags, y)
        else:
            with b.mark("denormalize_inv"):
                _denormalize(b, True, flags, y)
            with b.mark("horner_inv"):
                _horner(b, True, p, x, c1, c2, y[:p.WY + 1], scratch)
            with b.mark("lookup_inv"):
                b.del_lookup(table=packed, address=addr, target=target)
            with b.mark("normalize_inv"):
                _normalize(b, True, x, flags)

    compute(False)
    if not compute_only:
        F = p.WY + N_P
        with b.mark("phase"):
            for j in range(min(F, l0)):
                b.z_pow(y[j], Fraction(2) ** (j - F + 1))  # 2^(j - F) turns
        compute(True)
        for reg in (flags, y, c1, c2, scratch):
            b.free(reg)
    return b.finish_circuit(), (l0, l1, l2)


def check(p: P, n: int):
    circ, _ = build(p, compute_only=True)
    rng = np.random.default_rng(1)
    xs = np.unique(np.concatenate([rng.integers(1, MAX_X + 1, n), [1, 2, 3, 4, 5, 1023, 1024, 4095, 4096, MAX_X]]))
    want, _ = model(p, xs)
    sim = km.Simulator(batch_size=len(xs))
    sim.use_same_registers_as(circ)
    for i, v in enumerate(xs):
        sim.write_within_shot("x", i, int(v))
    sim.do(circ)
    got = np.array([sim.read_within_shot("y1", i, out=int) for i in range(len(xs))])
    bad = np.nonzero(got != want)[0]
    print(f"check: {len(xs)} x, mismatches {len(bad)}", flush=True)
    for i in bad[:5]:
        print("  x", xs[i], "circuit", got[i], "model", want[i])
    return len(bad) == 0


if __name__ == "__main__":
    out = sys.argv[1]
    kw = {k: int(v) for k, v in (a.split("=") for a in sys.argv[2:] if "=" in a)}
    p = P(**kw)
    if "--check" in sys.argv:
        if not check(p, int(sys.argv[sys.argv.index("--check") + 1])):
            sys.exit(1)
    circ, ws = build(p)
    circ.to_file(out, format="kmb")
    st = circ.operation_statistics()
    print(f"{p} widths {ws}: wrote {out}: {st.num_toffoli_like:,} Toffoli-like (static), {circ.num_qubits} qubits")
