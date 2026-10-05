r"""Does the agent's click alphabet even contain the winning clicks?

Mode 6 searches over a small set of candidate coordinates, chosen as the
centroids of rare, small objects. If a game's solution clicks somewhere that is
never offered, no amount of searching will find it - so compare the compressed
offline solution against the candidates the agent would actually generate.
"""
from __future__ import annotations

import argparse, json, logging, os, sys
from pathlib import Path

STARTER = Path(os.environ.get("ARC_STARTER", r"C:\prog\arc\starter"))
sys.path.insert(0, str(STARTER))
sys.path.insert(0, str(STARTER / "vendor" / "ARC-AGI-3-Agents"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction

from solve_offline import frame_of, quiet

ap = argparse.ArgumentParser()
ap.add_argument("--game", default="r11l")
ap.add_argument("--solutions", default=r"C:\prog\arc\solutions")
ap.add_argument("--take", type=int, default=10)
args = ap.parse_args()

quiet()
arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
env = arc.make(args.game)
obs = env.step(GameAction.RESET)
frame = frame_of(obs)

h, w = frame.shape
counts = np.bincount(frame.reshape(-1).astype(np.int32) & 0x1F, minlength=32)
bg = {int(c) for c in np.nonzero(counts > (h * w) // 8)[0]}
print(f"game={args.game}  background colours={sorted(bg)}")

seen = np.zeros((h, w), dtype=bool)
comps = []
for y in range(h):
    for x in range(w):
        if seen[y, x]:
            continue
        col = int(frame[y, x])
        if col in bg:
            seen[y, x] = True
            continue
        stack, cells = [(y, x)], []
        seen[y, x] = True
        while stack:
            cy, cx = stack.pop()
            cells.append((cy, cx))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and frame[ny, nx] == col:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        ys = [c[0] for c in cells]
        xs = [c[1] for c in cells]
        comps.append({"colour": col, "area": len(cells),
                      "cx": (min(xs) + max(xs)) // 2, "cy": (min(ys) + max(ys)) // 2,
                      "x0": min(xs), "x1": max(xs), "y0": min(ys), "y1": max(ys)})

# Same ranking the C++ core uses: rare colour first, then small shape.
comps.sort(key=lambda c: counts[c["colour"]] * 4 + c["area"])
alphabet = [(c["cx"], c["cy"]) for c in comps[:args.take]]
print(f"{len(comps)} objects; alphabet ({args.take}) = {alphabet}")

sol = Path(args.solutions) / f"{args.game}.json"
if not sol.exists():
    raise SystemExit(f"no solution for {args.game}")
d = json.loads(sol.read_text())
for L, traj in sorted(d.get("solutions", {}).items(), key=lambda kv: int(kv[0])):
    if int(L) == 0:
        continue
    clicks = [(x, y) for a, x, y in traj if a == 6]
    print(f"\nlevel {L}: {len(traj)} actions, clicks = {clicks}")
    for (x, y) in clicks:
        inside = [c for c in comps if c["x0"] <= x <= c["x1"] and c["y0"] <= y <= c["y1"]]
        tag = "IN alphabet" if (x, y) in alphabet else "not offered"
        if inside:
            c = inside[0]
            rank = comps.index(c)
            print(f"   ({x:2},{y:2}) {tag:12} -> object colour={c['colour']} area={c['area']} "
                  f"centroid=({c['cx']},{c['cy']}) rank={rank}")
        else:
            print(f"   ({x:2},{y:2}) {tag:12} -> not inside any object (background click)")
