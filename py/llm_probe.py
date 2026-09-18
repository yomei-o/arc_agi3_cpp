r"""Ask an LLM, once per game, what the game wants.

The point of this script is a single measurement: can an 8B model, shown one
board and the effect of each action, say what the player is supposed to do? If
it cannot, a 27B will not save it and the whole LLM direction is not worth its
40 seconds per call. So this writes the prompts and nothing else - running them
is a separate step on the build machine, where the model lives.

The prompt shows the start board in full and then, for each action, only the
cells that changed. Showing six whole boards would be six times the prefill for
information the diff already carries, and prefill is 71 t/s: a board costs about
half a second per hundred cells that never earned it.

    python llm_probe.py --out C:\tmp\prompts
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
from view_game import GLYPH, cell_size

MAX_DIFF = 10          # cells listed per action before it is summarised


def board(frame: np.ndarray, scale: int) -> np.ndarray:
    return frame[::scale, ::scale]


def render(cells: np.ndarray) -> str:
    return "\n".join("".join(GLYPH[int(v) % 16] for v in row) for row in cells)


def diff_lines(before: np.ndarray, after: np.ndarray) -> str:
    ys, xs = np.nonzero(before != after)
    if len(ys) == 0:
        return "no change"
    parts = []
    for y, x in list(zip(ys, xs))[:MAX_DIFF]:
        parts.append(f"({x},{y}) {GLYPH[int(before[y, x]) % 16]}->{GLYPH[int(after[y, x]) % 16]}")
    tail = "" if len(ys) <= MAX_DIFF else f" ... and {len(ys) - MAX_DIFF} more"
    return f"{len(ys)} cells changed: " + ", ".join(parts) + tail


def probe(game: str) -> str:
    quiet()
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    env = arc.make(game)
    obs = env.step(GameAction.RESET)
    frame = frame_of(obs)
    scale = cell_size(frame)
    start = board(frame, scale)

    counts = Counter(start.reshape(-1).tolist())
    legend = "  ".join(f"'{GLYPH[c % 16]}' = {n * 100 // start.size}% of the board"
                       for c, n in counts.most_common()[:6])

    avail = [a.value if hasattr(a, "value") else int(a) for a in obs.available_actions]
    trials = []
    for a in sorted(v for v in avail if v in (1, 2, 3, 4, 5)):
        e2 = arc.make(game)
        e2.step(GameAction.RESET)
        o = e2.step(BY_VALUE[a])
        f = frame_of(o)
        after = start if f is None else board(f, scale)
        trials.append((f"ACTION{a}", diff_lines(start, after)))

    if 6 in avail:
        h, w = start.shape
        for (cx, cy) in [(w // 2, h // 2), (w // 4, h // 4)]:
            e2 = arc.make(game)
            e2.step(GameAction.RESET)
            o = e2.step(BY_VALUE[6], data={"x": cx * scale, "y": cy * scale})
            f = frame_of(o)
            after = start if f is None else board(f, scale)
            trials.append((f"CLICK at cell ({cx},{cy})", diff_lines(start, after)))

    h, w = start.shape
    out = [
        f"You are shown one level of a puzzle video game on a {w}x{h} grid of cells.",
        "Each character is one cell colour.",
        "",
        f"Colour legend: {legend}",
        "",
        "BOARD:",
        render(start),
        "",
        "Effect of each available action, applied once from this board:",
    ]
    for name, d in trials:
        out.append(f"  {name}: {d}")
    out += [
        "",
        "Answer in at most three sentences, with no preamble:",
        "1. What does the player control?",
        "2. What must the player do to complete this level?",
    ]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="", help="comma list; default = every public game")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    quiet()
    if args.games:
        games = args.games.split(",")
    else:
        arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
        games = sorted(arc.list_games())

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for g in games:
        try:
            text = probe(g)
        except Exception as e:                       # one bad game must not stop the rest
            print(f"{g}: FAILED {type(e).__name__}: {e}")
            continue
        (out / f"{g}.txt").write_text(text, encoding="utf-8")
        print(f"{g}: {len(text)} chars")


if __name__ == "__main__":
    main()
