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
MAX_MOVE = 5           # cells an object may travel and still be called the same object

# Without the cap the matcher paired objects across the whole board, and on lf52
# - where the same 4-cell shape occurs a dozen times - it produced a chain of
# impossible journeys: one object "moving" to where the next one already was.
# Nothing that far away is evidence of a move; it is evidence of two objects.


def board(frame: np.ndarray, scale: int) -> np.ndarray:
    return frame[::scale, ::scale]


def render(cells: np.ndarray) -> str:
    return "\n".join("".join(GLYPH[int(v) % 16] for v in row) for row in cells)


def changed(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    return before != after


def components(cells: np.ndarray) -> list:
    """Every non-background 4-connected region as (colour, area, cx, cy)."""
    h, w = cells.shape
    counts = Counter(cells.reshape(-1).tolist())
    bg = {c for c, n in counts.items() if n * 8 > h * w}
    seen = np.zeros_like(cells, dtype=bool)
    out = []
    for y in range(h):
        for x in range(w):
            if seen[y, x] or int(cells[y, x]) in bg:
                continue
            col = int(cells[y, x])
            stack, pts = [(y, x)], []
            seen[y, x] = True
            while stack:
                cy, cx = stack.pop()
                pts.append((cy, cx))
                for ny, nx in ((cy-1, cx), (cy+1, cx), (cy, cx-1), (cy, cx+1)):
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and cells[ny, nx] == col:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            out.append((col, len(pts),
                        sum(q[1] for q in pts) // len(pts),
                        sum(q[0] for q in pts) // len(pts)))
    return out


def centroids(cells: np.ndarray, limit: int = 6) -> list:
    """Click targets: one point per object, rarest colour first.

    The first version of this probe clicked the middle of the board and a
    quarter of the way in. On ft09 both landed on bare background and the model
    was asked what the game wants having been shown that nothing does anything.
    That is not a test of the model.
    """
    h, w = cells.shape
    counts = Counter(cells.reshape(-1).tolist())
    ranked = sorted(components(cells), key=lambda o: (counts[o[0]], o[1]))
    picked = []
    for _, _, cx, cy in ranked:
        if all(abs(cx - a) + abs(cy - b) > 2 for a, b in picked):
            picked.append((cx, cy))
        if len(picked) >= limit:
            break
    return picked


def object_events(before: np.ndarray, after: np.ndarray, hud: np.ndarray = None) -> str:
    """Describe a step as things happening to objects, not as cells flipping.

    The cell-level version of this text got a cell-level answer from the model
    every time: shown two changed cells it said "change (6,9) to F", where the
    truth is that one sprite walked one square. It was never going to say
    anything else - the description it was given had no objects in it. So the
    matching is done here, where the board is, and the model is handed the
    result instead of the evidence for it.
    """
    if hud is not None:
        after = np.where(hud, before, after)
    if not (before != after).any():
        return "nothing happened"

    b, a = components(before), components(after)
    events, used = [], [False] * len(a)
    for col, area, cx, cy in b:
        best, bestd = -1, 1 << 30
        for i, (c2, a2, x2, y2) in enumerate(a):
            if used[i] or c2 != col or abs(a2 - area) * 4 > max(area, 1):
                continue
            d = abs(x2 - cx) + abs(y2 - cy)
            if d < bestd and d <= MAX_MOVE:
                best, bestd = i, d
        if best < 0:
            events.append(f"the colour-{GLYPH[col % 16]} object of {area} cells at ({cx},{cy}) disappeared")
            continue
        used[best] = True
        _, a2, x2, y2 = a[best]
        if (x2, y2) != (cx, cy):
            events.append(f"the colour-{GLYPH[col % 16]} object of {area} cells moved "
                          f"({x2 - cx:+d},{y2 - cy:+d}) from ({cx},{cy}) to ({x2},{y2})")
    for i, (col, area, cx, cy) in enumerate(a):
        if not used[i] and all(o[:1] != (col,) or (o[2], o[3]) != (cx, cy) for o in b):
            events.append(f"a colour-{GLYPH[col % 16]} object of {area} cells appeared at ({cx},{cy})")
    if not events:
        n = int((before != after).sum())
        return f"{n} cells changed colour but no object moved, appeared or vanished"
    return "; ".join(events[:6]) + ("" if len(events) <= 6 else f"; and {len(events) - 6} more")


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
        out.append(f"  {name}: {object_events(start, after, hud)}")
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
        games = sorted({e.game_id.split("-")[0] for e in arc.get_environments()})

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
