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


def changed(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    return before != after


def centroids(cells: np.ndarray, limit: int = 6) -> list:
    """Centroids of the non-background 4-connected regions, rarest colour first.

    The first version of this probe clicked the middle of the board and a
    quarter of the way in. On ft09 both landed on bare background and the model
    was asked what the game wants having been shown that nothing does anything.
    That is not a test of the model.
    """
    h, w = cells.shape
    counts = Counter(cells.reshape(-1).tolist())
    bg = {c for c, n in counts.items() if n * 8 > h * w}
    seen = np.zeros_like(cells, dtype=bool)
    out = []
    for y in range(h):
        for x in range(w):
            if seen[y, x] or int(cells[y, x]) in bg:
                continue
            col = cells[y, x]
            stack, pts = [(y, x)], []
            seen[y, x] = True
            while stack:
                cy, cx = stack.pop()
                pts.append((cy, cx))
                for ny, nx in ((cy-1, cx), (cy+1, cx), (cy, cx-1), (cy, cx+1)):
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and cells[ny, nx] == col:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            share = counts[int(col)] / float(h * w)
            out.append((share, len(pts),
                        sum(q[1] for q in pts) // len(pts), sum(q[0] for q in pts) // len(pts)))
    out.sort()
    picked = []
    for _, _, cx, cy in out:
        if all(abs(cx - a) + abs(cy - b) > 2 for a, b in picked):
            picked.append((cx, cy))
        if len(picked) >= limit:
            break
    return picked


def diff_lines(before: np.ndarray, after: np.ndarray, hud: np.ndarray = None) -> str:
    d = before != after
    if hud is not None:
        d = d & ~hud
    ys, xs = np.nonzero(d)
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
    h, w = start.shape

    counts = Counter(start.reshape(-1).tolist())
    legend = "  ".join(f"'{GLYPH[c % 16]}' = {n * 100 // start.size}% of the board"
                       for c, n in counts.most_common()[:6])

    avail = [a.value if hasattr(a, "value") else int(a) for a in obs.available_actions]

    def run(a, data=None):
        e2 = arc.make(game)
        e2.step(GameAction.RESET)
        o = e2.step(BY_VALUE[a], data=data or {})
        f = frame_of(o)
        return start if f is None else board(f, scale)

    trials = []
    for a in sorted(v for v in avail if v in (1, 2, 3, 4, 5)):
        trials.append((f"ACTION{a}", run(a)))
    if 6 in avail:
        for (cx, cy) in centroids(start):
            trials.append((f"CLICK on the object at ({cx},{cy})",
                           run(6, {"x": cx * scale, "y": cy * scale})))

    # Mask the cells that every action changes. On lf52 that is the whole of the
    # evidence: each action flips one cell at (0,0), a step counter, and the
    # first version of this prompt therefore asked the model what the game wants
    # while showing it nothing but a clock ticking.
    hud = None
    if trials:
        hud = np.ones_like(start, dtype=bool)
        for _, after in trials:
            hud &= changed(start, after)

    out = [
        f"You are shown one level of a puzzle video game on a {w}x{h} grid of cells.",
        "Each character is one cell colour. The actions are unlabelled buttons;",
        "their meaning is for you to infer. A scoreboard or step counter has been",
        "removed from the reported changes.",
        "",
        f"Colour legend: {legend}",
        "",
        "BOARD:",
        render(start),
        "",
        "Effect of each available action, applied once from this board:",
    ]
    for name, after in trials:
        out.append(f"  {name}: {diff_lines(start, after, hud)}")
    out += [
        "",
        "Answer in at most three sentences, with no preamble and no restating of",
        "the question. Be concrete: name the coordinates and colours you mean.",
        "1. What does the player control?",
        "2. What must the player do to complete this level?",
    ]
    return chr(10).join(out)


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
