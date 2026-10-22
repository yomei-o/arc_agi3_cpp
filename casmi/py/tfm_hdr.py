"""tfm_hdr: compare the config / fp3 vocabulary of TFM1 model files (v37_STATUS.md: production ft_all_ep5.bin vs the
holdout-safe ft8_ep5.bin used in the checks). usage: python tfm_hdr.py <a.bin> <b.bin> ...
"""
import hashlib
import math
import struct
import sys

ref = None
for p in sys.argv[1:]:
    with open(p, "rb") as f:
        magic, ver = struct.unpack("<ii", f.read(8))
        cfg = struct.unpack("<10i", f.read(40))
        lo, hi = struct.unpack("<ff", f.read(8))
        (nP,) = struct.unpack("<q", f.read(8))
        w = f.read(4 * nP)
        rE, rF, nv = struct.unpack("<iii", f.read(12))
        voc = f.read(8 * nv)
    ws = struct.unpack(f"<{min(nP, 1 << 20)}f", w[: 4 * min(nP, 1 << 20)])
    bad = sum(1 for x in ws if not math.isfinite(x))
    key = (cfg, lo, hi, nP, rE, rF, nv, hashlib.md5(voc).hexdigest())
    print(f"{p}: magic {magic:#x} v{ver} cfg(d L H ff demb HH NB maxPeaks nAd nIn)={cfg} lo/hi {lo}/{hi} nP {nP} "
          f"rE {rE} rF {rF} nVocab {nv} vocab md5 {key[-1][:12]} non-finite(first 1M w) {bad}")
    if ref is None:
        ref = key
    else:
        print("   same config + vocabulary as the first file:", key == ref)
