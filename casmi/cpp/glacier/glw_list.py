"""glw_list.py <file.bin> [all]: names and shapes of a GLW1 array file (glacier_w.bin / ref_*.bin)."""
import math, struct, sys

f = open(sys.argv[1], "rb")
assert f.read(4) == b"GLW1"
n, = struct.unpack("<I", f.read(4))
full = len(sys.argv) > 2
for _ in range(n):
    l, = struct.unpack("<I", f.read(4)); nm = f.read(l).decode()
    nd, = struct.unpack("<I", f.read(4)); dims = struct.unpack("<" + "I" * nd, f.read(4 * nd))
    f.seek(4 * math.prod(dims), 1)
    if not full and ".layers." in nm and ".layers.0." not in nm:
        continue
    print(nm, dims)
