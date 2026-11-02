"""Benchmark I circuit, own design v2 = v1 (inv_circuit.py) with
- binary shift count s (4 qubits): normalize by 16/8/4/2-bit stages, denormalize with one barrel rotation
  (left_rotate with a quantum shift), instead of 10 unary flags and 10 lane rotations each way;
- table lookup restricted to the reachable addresses (top two bits of x_norm are never 00);
- scratch only as wide as the widest Horner target.
Same arithmetic as inv_model.py.
    python inv_circuit2.py out.kmb [A=9 D=13 WY=30 W1=30 W2=28] [--check N]
"""
from __future__ import annotations
import sys
from fractions import Fraction
import numpy as np
import kickmix as km
from inv_model import P, N_P, WIDTH, MAX_X, fit_table, model
from inv_circuit import widths, _horner

S_BITS = 4  # s <= N_P = 10


def _zero_flag(b, x, k, flag):
    """flag ^= (top k bits of x are all zero)."""
    top = x[WIDTH - k:]
    b.x(top)
    b.multi_controlled_x(top, flag)
    b.x(top)


def _normalize(b, inverse, x, s):
    """x := x 2^(2 s), s binary (little endian)."""
    stages = list(reversed(range(S_BITS)))  # shift 2^(j+1) bits if the top 2^(j+1) bits are zero
    if not inverse:
        for j in stages:
            _zero_flag(b, x, 2 ** (j + 1), s[j])
            b.left_rotate(x, 2 ** (j + 1), control=s[j])
    else:
        for j in reversed(stages):
            b.right_rotate(x, 2 ** (j + 1), control=s[j])
            _zero_flag(b, x, 2 ** (j + 1), s[j])


def _lookup(b, inverse, packed, addr, target, lo):
    if inverse:
        b.del_lookup(table=packed, address=addr, target=target)
        return
    with b.unary_iteration(addr, range(lo, len(packed))) as it:
        for a, match in it:
            bits = [target[i] for i in range(len(target)) if (packed[a] >> i) & 1]
            if bits:
                b.cx(match, km.array(bits))


def build(p: P, compute_only=False):
    table = fit_table(p)
    l0, l1, l2 = widths(p, table)
    packed = [int(c0) | (int(c1) << l0) | (int(c2) << (l0 + l1)) for c0, c1, c2 in table]
    lo = 2 ** (p.A - 2)
    b = km.CircuitBuilder()
    x = b.create_quantum_register(WIDTH, name="x")
    alloc = (lambda n, nm: b.create_quantum_register(n, name=nm)) if compute_only else (lambda n, nm: b.alloc_qubits(n))
    s = alloc(S_BITS, "s")
    y = alloc(l0, "y1")
    c1 = alloc(l1, "c1")
    c2 = alloc(l2, "c2")
    scratch = alloc(p.WY + 1, "scratch")
    addr = x[WIDTH - p.A:]
    target = km.array(list(y) + list(c1) + list(c2))

    def compute(inverse):
        if not inverse:
            _normalize(b, False, x, s)
            _lookup(b, False, packed, addr, target, lo)
            _horner(b, False, p, x, c1, c2, y[:p.WY + 1], scratch)
            b.left_rotate(y, s)
        else:
            b.right_rotate(y, s)
            _horner(b, True, p, x, c1, c2, y[:p.WY + 1], scratch)
            _lookup(b, True, packed, addr, target, lo)
            _normalize(b, True, x, s)

    compute(False)
    if not compute_only:
        F = p.WY + N_P
        for j in range(min(F, l0)):
            b.z_pow(y[j], Fraction(2) ** (j - F + 1))  # 2^(j - F) turns
        compute(True)
        for reg in (s, y, c1, c2, scratch):
            b.free(reg)
    return b.finish_circuit(), (l0, l1, l2)


def check(p: P, n: int):
    circ, _ = build(p, compute_only=True)
    rng = np.random.default_rng(1)
    xs = np.unique(np.concatenate([rng.integers(1, MAX_X + 1, n), np.arange(1, 70), [1023, 1024, 4095, 4096, 65535, 65536, MAX_X]]))
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
