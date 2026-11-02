"""Benchmark I circuit, own design v3 (lazy register allocation) / v2 = v1 (inv_circuit.py) with
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
    packed = [int(c0) | (int(c1) << (p.WY + 1)) | (int(c2) << (p.WY + 1 + l1)) for c0, c1, c2 in table]
    lo = 2 ** (p.A - 2)
    b = km.CircuitBuilder()
    x = b.create_quantum_register(WIDTH, name="x")
    named = lambda n, nm: b.create_quantum_register(n, name=nm)
    addr = x[WIDTH - p.A:]
    ly = p.WY + 1
    # v3: registers live only while needed, so the builder reuses freed qubits for the temporaries
    s = named(S_BITS, "s") if compute_only else b.alloc_qubits(S_BITS)
    _normalize(b, False, x, s)
    y_lo = named(ly, "y_lo") if compute_only else b.alloc_qubits(ly)
    c1, c2 = b.alloc_qubits(l1), b.alloc_qubits(l2)
    target = km.array(list(y_lo) + list(c1) + list(c2))
    _lookup(b, False, packed, addr, target, lo)
    scratch = b.alloc_qubits(ly)
    _horner(b, False, p, x, c1, c2, y_lo, scratch)
    b.free(scratch)
    y_hi = named(l0 - ly, "y_hi") if compute_only else b.alloc_qubits(l0 - ly)
    y = km.array(list(y_lo) + list(y_hi))
    b.left_rotate(y, s)
    if not compute_only:
        F = p.WY + N_P
        for j in range(min(F, l0)):
            b.z_pow(y[j], Fraction(2) ** (j - F + 1))  # 2^(j - F) turns
        b.right_rotate(y, s)
        b.free(y_hi)
        scratch = b.alloc_qubits(ly)
        _horner(b, True, p, x, c1, c2, y_lo, scratch)
        b.free(scratch)
        _lookup(b, True, packed, addr, target, lo)
        b.free(km.array(list(y_lo) + list(c1) + list(c2)))
        _normalize(b, True, x, s)
        b.free(s)
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
    got = np.array([sim.read_within_shot("y_lo", i, out=int) + (sim.read_within_shot("y_hi", i, out=int) << (p.WY + 1)) for i in range(len(xs))])
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
