r"""Measure what a local simulator costs us.

The whole offline plan rests on being able to snapshot a game, explore from it,
and come back. Three numbers decide the algorithm:
  * step rate            - how fast we can advance the real game
  * deepcopy cost/size   - whether we can keep an archive of restorable states
  * replay cost          - the fallback when an archive entry is only a
                           trajectory (the games are deterministic, so replaying
                           the action list from RESET reproduces a state exactly)
"""
from __future__ import annotations

import argparse, copy, logging, os, pickle, sys, time
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
ap.add_argument("--n", type=int, default=200)
args = ap.parse_args()

logging.getLogger().setLevel(logging.ERROR)
for n in ("arc_agi", "arc_agi.scorecard", "root"):
    logging.getLogger(n).setLevel(logging.ERROR)

arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
env = arc.make(args.game)
env.step(GameAction.RESET)

acts = [a for a in env.action_space if not a.is_complex()] or [GameAction.ACTION1]

t0 = time.time()
for i in range(args.n):
    env.step(acts[i % len(acts)])
step_ms = (time.time() - t0) * 1000 / args.n
print(f"step:            {step_ms:7.3f} ms  ({1000 / step_ms:.0f}/s)")

t0 = time.time()
snaps = [copy.deepcopy(env) for _ in range(20)]
copy_ms = (time.time() - t0) * 1000 / 20
print(f"deepcopy:        {copy_ms:7.3f} ms")

# Rough resident size of one snapshot. The game object holds numpy arrays and
# sprite tables; pickling the *game* is what an archive would really hold.
import tracemalloc
tracemalloc.start()
base = tracemalloc.get_traced_memory()[0]
hold = [copy.deepcopy(env) for _ in range(50)]
peak = tracemalloc.get_traced_memory()[0]
tracemalloc.stop()
print(f"snapshot size:   {(peak - base) / 50 / 1024:7.1f} KiB each "
      f"({(peak - base) / 1024 / 1024:.1f} MiB for 50)")
del hold

# Determinism + restore fidelity. Reset first: a long random prefix can leave
# the game in GAME_OVER, where responses legitimately carry no frame.
env.step(GameAction.RESET)
for i in range(25):
    env.step(acts[i % len(acts)])
a = copy.deepcopy(env)
b = copy.deepcopy(env)
def roll(e):
    out = []
    for i in range(30):
        o = e.step(acts[i % len(acts)])
        out.append(np.asarray(o.frame[-1]) if o and o.frame else None)
    return out


fa, fb = roll(a), roll(b)
same = all((x is None and y is None) or (x is not None and y is not None and (x == y).all()) for x, y in zip(fa, fb))
print(f"restore fidelity: {'identical' if same else 'DIVERGED'}"
      f" over 30 steps from two copies of the same state")

# Replay-from-RESET cost, the memory-light alternative to keeping snapshots.
t0 = time.time()
e2 = arc.make(args.game)
e2.step(GameAction.RESET)
for i in range(100):
    e2.step(acts[i % len(acts)])
replay_ms = (time.time() - t0) * 1000
print(f"replay 100 steps:{replay_ms:7.1f} ms  (archive entry restore cost at depth 100)")

print(f"\nbudget check: a 1M-step Go-Explore run costs "
      f"{args.n and 1_000_000 * step_ms / 1000 / 60:.1f} core-minutes at this step rate")
