r"""Check the state abstraction the offline solver relies on.

Go-Explore only works if distinct game states hash to distinct keys. Two ways
that breaks: the HUD mask covers so much that everything collapses to one key,
or it covers so little that the step counter makes every state unique. This
prints both numbers so the archive size stops being a mystery.
"""
from __future__ import annotations

import argparse, os, random, sys
from pathlib import Path

STARTER = Path(os.environ.get("ARC_STARTER", r"C:\prog\arc\starter"))
sys.path.insert(0, str(STARTER))
sys.path.insert(0, str(STARTER / "vendor" / "ARC-AGI-3-Agents"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction

from solve_offline import hud_mask, frame_of, quiet

ap = argparse.ArgumentParser()
ap.add_argument("--game", default="ls20")
ap.add_argument("--steps", type=int, default=400)
args = ap.parse_args()

quiet()
arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)

mask = hud_mask(arc, args.game)
print(f"game={args.game}")
print(f"mask covers {mask.sum()} / {mask.size} cells ({100*mask.sum()/mask.size:.1f}%)")

env = arc.make(args.game)
obs = env.step(GameAction.RESET)
acts = [a for a in env.action_space if not a.is_complex()]
rng = random.Random(7)

raw, masked = set(), set()
changed_cells = np.zeros((64, 64), dtype=np.int32)
prev = frame_of(obs)
for i in range(args.steps):
    if acts:
        o = env.step(rng.choice(acts))
    else:
        o = env.step(GameAction.ACTION6, data={"x": rng.randrange(64), "y": rng.randrange(64)})
    f = frame_of(o)
    if f is None:
        o = env.step(GameAction.RESET)
        f = frame_of(o)
        if f is None:
            break
    changed_cells += (f != prev)
    prev = f
    raw.add(f.tobytes())
    masked.add(np.where(mask, np.int8(0), f).tobytes())

print(f"over {args.steps} random steps: {len(raw)} distinct raw frames, "
      f"{len(masked)} distinct masked frames")
print(f"cells that ever changed: {(changed_cells > 0).sum()}")
print(f"cells that changed on >90% of steps (clock-like): {(changed_cells > args.steps*0.9).sum()}")
inform = (changed_cells > 0) & (~mask)
print(f"cells that change AND survive the mask: {inform.sum()}")
if inform.sum():
    ys, xs = np.nonzero(inform)
    print(f"  their bbox: x {xs.min()}..{xs.max()}, y {ys.min()}..{ys.max()}")
