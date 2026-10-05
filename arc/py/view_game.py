r"""Look at a game the way a person would.

The 64x64 frame is a rendered view: the logical board is drawn several pixels
per cell with HUD furniture around it. Printing the raw frame is unreadable, so
this finds the pixel scale from where the colour changes and prints one
character per logical cell, which is small enough to actually read.

    python view_game.py --game ka59 --actions 1,1,2,5
"""
from __future__ import annotations

import argparse, os, sys
from collections import Counter
from pathlib import Path

STARTER = Path(os.environ.get("ARC_STARTER", r"C:\prog\arc\starter"))
sys.path.insert(0, str(STARTER))
sys.path.insert(0, str(STARTER / "vendor" / "ARC-AGI-3-Agents"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction

from solve_offline import BY_VALUE, frame_of, quiet

GLYPH = ".:-=+*#%@ABCDEFG"   # colour 0..15


def cell_size(frame: np.ndarray) -> int:
    """Pixels per logical cell, from the spacing of colour changes."""
    edges = set()
    for y in range(frame.shape[0]):
        row = frame[y]
        edges.update(x + 1 for x in np.nonzero(row[:-1] != row[1:])[0])
    for x in range(frame.shape[1]):
        col = frame[:, x]
        edges.update(y + 1 for y in np.nonzero(col[:-1] != col[1:])[0])
    if not edges:
        return 1
    best, best_hits = 1, -1
    for s in range(8, 1, -1):
        hits = sum(1 for e in edges if e % s == 0)
        if hits > best_hits * 1.05:      # prefer the largest scale that explains most edges
            best, best_hits = s, hits
    return best


def show(frame: np.ndarray, title: str, scale: int) -> None:
    h, w = frame.shape
    print(f"\n--- {title}  (scale {scale}px/cell) ---")
    header = "   " + "".join(str((x // scale) % 10) for x in range(0, w, scale))
    print(header)
    for y in range(0, h, scale):
        row = "".join(GLYPH[int(frame[y, x]) % 16] for x in range(0, w, scale))
        print(f"{y // scale:2} {row}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="ka59")
    ap.add_argument("--actions", default="", help="comma list, e.g. 1,1,2 or 6:32:18")
    ap.add_argument("--every", action="store_true", help="print a frame after each action")
    args = ap.parse_args()

    quiet()
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    env = arc.make(args.game)
    obs = env.step(GameAction.RESET)
    frame = frame_of(obs)
    scale = cell_size(frame)

    counts = Counter(frame.reshape(-1).tolist())
    print(f"game={args.game}  actions={obs.available_actions}  win_levels={obs.win_levels}")
    print("colours (share of board): " + "  ".join(
        f"{GLYPH[c % 16]}={c}:{n * 100 // frame.size}%" for c, n in counts.most_common()))
    show(frame, "start", scale)

    if not args.actions:
        return
    for token in args.actions.split(","):
        parts = token.split(":")
        a = int(parts[0])
        data = {"x": int(parts[1]), "y": int(parts[2])} if len(parts) == 3 else {}
        obs = env.step(BY_VALUE[a], data=data)
        frame = frame_of(obs)
        if frame is None:
            print(f"\nafter {token}: no frame (state={obs.state if obs else '?'})")
            break
        if args.every:
            show(frame, f"after {token}  levels={obs.levels_completed} state={obs.state}", scale)
    if not args.every:
        show(frame, f"after {args.actions}  levels={obs.levels_completed}", scale)


if __name__ == "__main__":
    main()
