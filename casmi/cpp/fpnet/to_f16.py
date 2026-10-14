# fpnet_full1.bin (FPN1, f32) -> FPN2 (same layout; 2-D weight matrices f16, 1-D tensors f32); fpnet_ggml widens to f32 at load
# (1-D must stay f32: SinEmb inv frequencies reach 628, so f16 rounding = ~200 rad phase error at m/z 1000)
# usage: to_f16.py in.bin out.bin
import sys, struct, numpy as np
b = open(sys.argv[1], "rb").read()
assert b[:4] == b"FPN1"
hdr = struct.unpack_from("<5i", b, 4); p = 24
out = [b"FPN2", struct.pack("<5i", *hdr)]
maxrel = 0.0
for _ in range(hdr[4]):
    nl, = struct.unpack_from("<i", b, p); p += 4
    name = b[p:p + nl]; p += nl
    nd, = struct.unpack_from("<i", b, p); p += 4
    dims = struct.unpack_from(f"<{nd}i", b, p); p += 4 * nd
    n = int(np.prod(dims))
    a = np.frombuffer(b, "<f4", n, p); p += 4 * n
    h = a.astype("<f2") if nd == 2 else a.astype("<f4")  # 1-D (biases, norms, SinEmb inv freqs up to 628) stay f32
    assert np.isfinite(h).all(), name
    out += [struct.pack("<i", nl), name, struct.pack("<i", nd), struct.pack(f"<{nd}i", *dims), h.tobytes()]
assert p == len(b)
open(sys.argv[2], "wb").write(b"".join(out))
print("ok", hdr)
