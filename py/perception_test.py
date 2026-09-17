r"""Show what each action actually does to a game's frame.

Used to design the perception side of the C++ core: how big the frames' static
background is, whether anything animates on its own, and how the controllable
sprite responds to each action.
"""
from __future__ import annotations

import argparse, logging, os, sys
from collections import Counter
from pathlib import Path

STARTER = Path(os.environ.get("ARC_STARTER", r"C:\prog\arc\starter"))
sys.path.insert(0, str(STARTER))
sys.path.insert(0, str(STARTER / "vendor" / "ARC-AGI-3-Agents"))

import numpy as np
import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction

ap = argparse.ArgumentParser()
ap.add_argument("--game", default="ls20")
ap.add_argument("--show", action="store_true", help="print the frame as ASCII")
ap.add_argument("--noop-steps", type=int, default=5,
                help="how many times to repeat one action to spot self-animation")
args = ap.parse_args()

logging.getLogger().setLevel(logging.ERROR)
for n in ("arc_agi", "arc_agi.scorecard", "root"):
    logging.getLogger(n).setLevel(logging.ERROR)

arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
env = arc.make(args.game)
obs = env.step(GameAction.RESET)
g0 = np.asarray(obs.frame[-1])

print(f"game={args.game}  available_actions={obs.available_actions}  win_levels={obs.win_levels}")
print(f"frame shape={g0.shape}  colour histogram={dict(Counter(g0.flatten().tolist()).most_common())}")

PALETTE = " .:-=+*#%@ABCDEF"


def show(g, title):
    print(f"\n--- {title} ---")
    for row in g:
        print("".join(PALETTE[int(v) % 16] for v in row))


if args.show:
    show(g0, "initial frame")


def describe(a, b, label):
    d = np.argwhere(a != b)
    if len(d) == 0:
        print(f"  {label:22} -> NO CHANGE")
        return
    ys, xs = d[:, 0], d[:, 1]
    pairs = Counter((int(a[y, x]), int(b[y, x])) for y, x in d)
    print(f"  {label:22} -> {len(d):4} cells changed, "
          f"bbox=(x {xs.min()}..{xs.max()}, y {ys.min()}..{ys.max()}), "
          f"transitions={dict(list(pairs.most_common(6)))}")


print("\n=== repeating one action (does the world move on its own?) ===")
prev = g0
for i in range(args.noop_steps):
    obs = env.step(GameAction.ACTION1)
    cur = np.asarray(obs.frame[-1])
    describe(prev, cur, f"ACTION1 #{i + 1}")
    prev = cur

print("\n=== each action once, from a fresh reset ===")
for act in obs.available_actions if obs.available_actions else [1, 2, 3, 4, 5]:
    a = GameAction(act) if not isinstance(act, GameAction) else act
    o = env.step(GameAction.RESET)
    base = np.asarray(o.frame[-1])
    data = {"x": 32, "y": 32} if a.is_complex() else {}
    o2 = env.step(a, data=data)
    describe(base, np.asarray(o2.frame[-1]), a.name)

print("\n=== same action twice in a row (is movement repeatable?) ===")
o = env.step(GameAction.RESET)
prev = np.asarray(o.frame[-1])
for i in range(6):
    o = env.step(GameAction.ACTION2)
    cur = np.asarray(o.frame[-1])
    describe(prev, cur, f"ACTION2 #{i + 1}")
    prev = cur
if args.show:
    show(prev, "after 6x ACTION2")
