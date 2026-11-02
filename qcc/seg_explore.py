"""How many table segments of degree d on M in [1,4) give 1/sqrt(M) to relative 2^-26 (no Newton step)?"""
import numpy as np
for d in (1, 2, 3, 4):
    for k in range(3, 11):
        n = 3 * 2**k // 4 * 1  # segments over [1,4) of width 2^-(k-2)... uniform width w
        w = 4.0 / 2**k          # top k bits of x_norm (22 bits) -> M width 4/2^k; segments below M=1 unused
        worst = 0.0
        for a in np.arange(1.0, 4.0, w):
            t = np.cos((2 * np.arange(d + 1) + 1) * np.pi / (2 * d + 2))
            xs = a + w / 2 * (1 + t)
            c = np.polyfit(xs - a, 1 / np.sqrt(xs), d)
            xx = np.linspace(a, a + w, 200)
            e = np.abs(np.polyval(c, xx - a) * np.sqrt(xx) - 1).max()
            worst = max(worst, e)
        print(f"d={d} addr_bits={k} segments={int(3 * 2**k / 4)} max rel err 2^{np.log2(worst):.1f}")
