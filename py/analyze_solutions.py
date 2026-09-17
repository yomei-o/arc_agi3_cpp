r"""Turn solved trajectories into an answer to "what finishes a level?".

The offline solver produces the shortest action sequence it found to each level
of each public game. On its own that is not useful on Kaggle, where the games
are different ones. What transfers is the ANSWER TO A QUESTION: at the moment a
level completes, what was true of the state and the action?

So this replays each solution, stops on the step where levels_completed goes up,
and records generic, game-independent facts about that moment - which action,
what the avatar was standing on, how rare that colour was, whether the object
had been touched before, how the frame changed. Aggregated over 25 games those
facts are what a transferable policy can be trained on, and in the meantime they
tell us directly which heuristics are worth putting in the C++ core.
"""
from __future__ import annotations

import argparse, json, os, sys
from collections import Counter, defaultdict
from pathlib import Path

STARTER = Path(os.environ.get("ARC_STARTER", r"C:\prog\arc\starter"))
sys.path.insert(0, str(STARTER))
sys.path.insert(0, str(STARTER / "vendor" / "ARC-AGI-3-Agents"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction

from solve_offline import BY_VALUE, frame_of, object_centroids, quiet


def component_of(frame: np.ndarray, y: int, x: int) -> tuple[int, list[tuple[int, int]]]:
    """The same-colour region containing (y, x)."""
    col = int(frame[y, x])
    h, w = frame.shape
    seen = {(y, x)}
    stack = [(y, x)]
    cells = []
    while stack:
        cy, cx = stack.pop()
        cells.append((cy, cx))
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w and (ny, nx) not in seen and frame[ny, nx] == col:
                seen.add((ny, nx))
                stack.append((ny, nx))
    return col, cells


def moving_shape(a: np.ndarray, b: np.ndarray) -> tuple[int, int, int] | None:
    """(colour, y, x) of the smallest shape that translated between two frames."""
    diff = np.argwhere(a != b)
    if len(diff) == 0 or len(diff) > 400:
        return None
    best = None
    for col in set(int(a[y, x]) for y, x in diff):
        gone = [(y, x) for y, x in diff if a[y, x] == col]
        came = [(y, x) for y, x in diff if b[y, x] == col]
        if len(gone) != len(came) or not gone:
            continue
        dy, dx = came[0][0] - gone[0][0], came[0][1] - gone[0][1]
        if (dy, dx) == (0, 0):
            continue
        if all(0 <= y + dy < a.shape[0] and 0 <= x + dx < a.shape[1] and b[y + dy, x + dx] == col
               for y, x in gone):
            if best is None or len(gone) < best[0]:
                best = (len(gone), col, came[0][0], came[0][1])
    return (best[1], best[2], best[3]) if best else None


def analyse(game: str, sol: dict) -> list[dict]:
    quiet()
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    out = []
    solutions = sol.get("solutions", {})
    # The deepest solution contains every shallower one as a prefix.
    if not solutions:
        return out
    deepest = max(solutions, key=lambda k: int(k))
    traj = solutions[deepest]

    env = arc.make(game)
    obs = env.step(GameAction.RESET)
    prev_frame = frame_of(obs)
    prev_levels = 0
    avatar = None  # (colour, y, x)
    touched_colors: Counter[int] = Counter()

    for i, (a, x, y) in enumerate(traj):
        act = BY_VALUE[a]
        obs = env.step(act, data={"x": x, "y": y} if act.is_complex() else {})
        frame = frame_of(obs)
        if frame is None:
            continue
        levels = int(obs.levels_completed)

        mv = moving_shape(prev_frame, frame)
        if mv is not None:
            avatar = mv

        if levels > prev_levels:
            rec: dict = {"game": game, "level": prev_levels, "action_index": i,
                         "action": act.name, "actions_used_total": i + 1}
            if act.is_complex():
                rec["click"] = [x, y]
                col, cells = component_of(prev_frame, y, x)
                rec["clicked_colour"] = col
                rec["clicked_area"] = len(cells)
                counts = Counter(prev_frame.reshape(-1).tolist())
                rec["clicked_colour_share"] = round(counts[col] / prev_frame.size, 4)
            if avatar is not None:
                acol, ay, ax = avatar
                rec["avatar_colour"] = acol
                under = int(prev_frame[min(ay, 63), min(ax, 63)])
                rec["standing_on"] = under
                counts = Counter(prev_frame.reshape(-1).tolist())
                rec["standing_on_share"] = round(counts[under] / prev_frame.size, 4)
                rec["standing_on_is_rare"] = counts[under] < prev_frame.size * 0.02
            changed = int((prev_frame != frame).sum())
            rec["cells_changed"] = changed
            rec["whole_board_repaint"] = changed > prev_frame.size * 0.3
            out.append(rec)
            prev_levels = levels
        prev_frame = frame
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solutions", default="solutions")
    ap.add_argument("--out", default="level_completions.json")
    args = ap.parse_args()

    recs = []
    files = sorted(Path(args.solutions).glob("*.json"))
    if not files:
        raise SystemExit(f"no solution files under {args.solutions}")
    for f in files:
        sol = json.loads(f.read_text())
        if sol.get("max_levels", 0) < 1:
            continue
        try:
            recs += analyse(sol["game"], sol)
        except Exception as e:
            print(f"  {sol['game']}: {type(e).__name__}: {e}")

    Path(args.out).write_text(json.dumps(recs, indent=1))
    print(f"{len(recs)} level completions from {len(files)} games -> {args.out}\n")

    if not recs:
        return
    print("action that completed the level:")
    for k, v in Counter(r["action"] for r in recs).most_common():
        print(f"   {k:10} {v}")
    on_rare = [r for r in recs if r.get("standing_on_is_rare")]
    print(f"\ncompleted while the avatar stood on a rare colour: "
          f"{len(on_rare)}/{sum(1 for r in recs if 'standing_on' in r)}")
    repaint = sum(1 for r in recs if r.get("whole_board_repaint"))
    print(f"completions that repainted most of the board (level transition): {repaint}/{len(recs)}")
    by_game = defaultdict(list)
    for r in recs:
        by_game[r["game"]].append(r["level"])
    print(f"\ngames with at least one completion: {len(by_game)}")


if __name__ == "__main__":
    main()
