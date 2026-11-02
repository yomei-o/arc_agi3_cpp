"""Static Toffoli count per stage of inv_circuit: builds the stage prefixes and prints the differences."""
import sys
import kickmix as km
from nr_reference_circuit import _normalize, _denormalize
from inv_model import P, N_P, WIDTH, fit_table
from inv_circuit import widths, _horner

p = P(**{k: int(v) for k, v in (a.split("=") for a in sys.argv[1:])})
table = fit_table(p)
l0, l1, l2 = widths(p, table)
packed = [int(c0) | (int(c1) << l0) | (int(c2) << (l0 + l1)) for c0, c1, c2 in table]
NAMES = ["normalize", "lookup", "horner", "denormalize", "denormalize_inv", "horner_inv", "lookup_inv", "normalize_inv"]


def count(k):
    b = km.CircuitBuilder()
    x = b.create_quantum_register(WIDTH, name="x")
    flags, y, c1, c2, scratch = (b.alloc_qubits(n) for n in (N_P, l0, l1, l2, l0))
    addr, target = x[WIDTH - p.A:], km.array(list(y) + list(c1) + list(c2))
    st = [lambda: _normalize(b, False, x, flags),
          lambda: b.init_lookup(table=packed, address=addr, target=target),
          lambda: _horner(b, False, p, x, c1, c2, y[:p.WY + 1], scratch),
          lambda: _denormalize(b, False, flags, y),
          lambda: _denormalize(b, True, flags, y),
          lambda: _horner(b, True, p, x, c1, c2, y[:p.WY + 1], scratch),
          lambda: b.del_lookup(table=packed, address=addr, target=target),
          lambda: _normalize(b, True, x, flags)]
    for f in st[:k]:
        f()
    return b.finish_circuit().operation_statistics().num_toffoli_like


prev = 0
for k, n in enumerate(NAMES, 1):
    c = count(k)
    print(f"{n:16s} {c - prev:6d}")
    prev = c
print(f"{'total':16s} {prev:6d}   widths {l0, l1, l2}")
