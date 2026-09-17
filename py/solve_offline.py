r"""Offline solver: find short winning action sequences for the public games.

Locally the games arrive as Python classes, so `copy.deepcopy(env)` gives a
perfect simulator and a restored state replays identically (verified in
bench_sim.py). That turns each public game into a search problem, and the search
we want is Go-Explore: keep an archive of distinct states, return to a promising
one, explore from it, and - crucially - whenever a state is reached again by a
shorter route, keep the shorter one.

That last rule is why this is the right algorithm here rather than a generic
planner: the competition scores a completed level as (human/agent actions)^2, so
what we need from the offline phase is not merely a solution but a SHORT one.
Go-Explore optimises exactly that quantity as a side effect of its archive.

The output - shortest trajectory to each level of each public game, together
with the frames along it - is the training material for the transferable policy
that has to play the hidden games without a simulator.

Two environment facts this relies on, both established by reading arcengine:
  * mid-game RESET restarts only the CURRENT level and keeps levels_completed
    (base_game.handle_reset). A RESET when action_count == 0 is a full reset.
  * the frame carries HUD furniture - a remaining-actions bar - that advances
    every step regardless of what we do. Hashing the raw frame would make every
    state novel, so the HUD is detected and masked out before hashing.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path

STARTER = Path(os.environ.get("ARC_STARTER", r"C:\prog\arc\starter"))
sys.path.insert(0, str(STARTER))
sys.path.insert(0, str(STARTER / "vendor" / "ARC-AGI-3-Agents"))

import numpy as np
import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction, GameState

SIMPLE = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
          GameAction.ACTION4, GameAction.ACTION5, GameAction.ACTION7]
BY_VALUE = {a.value: a for a in GameAction}


def quiet() -> None:
    logging.getLogger().setLevel(logging.ERROR)
    for n in ("arc_agi", "arc_agi.scorecard", "root", "arc_agi.base"):
        logging.getLogger(n).setLevel(logging.ERROR)


# ---------------------------------------------------------------------------
# Perception helpers
# ---------------------------------------------------------------------------
def safe_step(env, action: GameAction, data: dict | None = None):
    """Step, tolerating a game that throws.

    Some games raise from inside their own logic on particular actions (seen:
    "qoljprchpbb not attached to oegtnpbqims" on an ACTION6). A search that
    explores every action will find those, and one of them must not end the run.
    """
    try:
        return env.step(action, data=data or {})
    except Exception:
        return None


def frame_of(obs) -> np.ndarray | None:
    if obs is None or not obs.frame:
        return None
    return np.asarray(obs.frame[-1], dtype=np.int8)


def hud_mask(fresh, rollouts: int = 4, length: int = 80) -> np.ndarray:
    """Clock cells: they advance with the step index whatever we do.

    Run several *different* random rollouts of equal length from a full reset
    and keep the cells that (a) hold the same value at every step across all of
    them, and (b) actually change over the course of a rollout. Condition (b)
    matters more than it looks: without it the mask also swallows every cell
    that random play merely failed to disturb - a pushable block, a door, a
    switch - and those are exactly the cells that distinguish one state from
    another. Masking them makes the whole archive collapse to "where the avatar
    is", which is how this search first came back with 22 states after 30k
    steps.

    Static scenery is deliberately left unmasked. It costs nothing in the hash
    while it stays constant, and the moment the game does change it we want to
    notice.
    """
    seqs = []
    for r in range(rollouts):
        env = fresh()
        rng = random.Random(1000 + r)
        acts = [a for a in env.action_space if not a.is_complex()]
        frames = []
        for _ in range(length):
            if acts:
                o = safe_step(env, rng.choice(acts))
            else:
                o = safe_step(env, GameAction.ACTION6,
                              {"x": rng.randrange(64), "y": rng.randrange(64)})
            f = frame_of(o)
            if f is None:
                break
            frames.append(f)
        seqs.append(frames)

    n = min(len(s) for s in seqs)
    if n == 0:
        return np.zeros((64, 64), dtype=bool)
    same = np.ones((64, 64), dtype=bool)
    varies = np.zeros((64, 64), dtype=bool)
    for t in range(n):
        for r in range(len(seqs)):
            if r:
                same &= seqs[r][t] == seqs[0][t]
            varies |= seqs[r][t] != seqs[r][0]
    return same & varies


def action_candidates(env, frame: np.ndarray, rng: random.Random, max_clicks: int = 16):
    """Legal actions, with sensible targets for the coordinate action.

    A 64x64 click space would swamp the search, so object centroids are offered
    as a prior - but only as a prior. On FT09 a centroid-only candidate set
    produced an archive of ONE state in 30k steps while uniform random clicking
    reached 14 distinct frames in 300, because what that game responds to is not
    the middle of a shape. Centroids plus random coordinates keeps the useful
    bias without inheriting its blind spot.
    """
    out = []
    complex_ok = False
    for a in env.action_space:
        if a.is_complex():
            complex_ok = True
        else:
            out.append((a.value, 0, 0))
    if complex_ok:
        if frame is not None:
            for (x, y) in object_centroids(frame)[:max_clicks]:
                out.append((6, x, y))
        for _ in range(max_clicks):
            out.append((6, rng.randrange(64), rng.randrange(64)))
    return out


def object_centroids(frame: np.ndarray) -> list[tuple[int, int]]:
    """Centroids of same-colour connected components, largest first.

    Deliberately simple: colours covering more than an eighth of the board are
    scenery and skipped.
    """
    h, w = frame.shape
    counts = np.bincount(frame.reshape(-1).astype(np.int32) & 0x1F, minlength=32)
    bg = {int(c) for c in np.nonzero(counts > (h * w) // 8)[0]}
    seen = np.zeros((h, w), dtype=bool)
    out = []
    for y in range(h):
        for x in range(w):
            if seen[y, x]:
                continue
            col = int(frame[y, x])
            if col in bg:
                seen[y, x] = True
                continue
            stack = [(y, x)]
            seen[y, x] = True
            cells = []
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
            out.append((len(cells), (min(xs) + max(xs)) // 2, (min(ys) + max(ys)) // 2))
    out.sort(reverse=True)
    return [(x, y) for _, x, y in out]


# ---------------------------------------------------------------------------
# Go-Explore
# ---------------------------------------------------------------------------
class Entry:
    __slots__ = ("key", "traj", "levels", "visits", "chosen", "snap")

    def __init__(self, key: bytes, traj: bytes, levels: int) -> None:
        self.key = key
        self.traj = traj
        self.levels = levels
        self.visits = 0
        self.chosen = 0
        self.snap = None


def pack(traj: list[tuple[int, int, int]]) -> bytes:
    b = bytearray()
    for a, x, y in traj:
        b += bytes((a, x, y))
    return bytes(b)


def unpack(b: bytes) -> list[tuple[int, int, int]]:
    return [(b[i], b[i + 1], b[i + 2]) for i in range(0, len(b), 3)]


class Solver:
    def __init__(self, game: str, seed: int = 0, snap_cache: int = 300) -> None:
        quiet()
        self.game = game
        self.arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
        self.rng = random.Random(seed)
        # Build the environment exactly once. arc.make() contacts the ARC API,
        # and calling it per restore had 14 workers hammering the service into
        # SSL failures; a deepcopy of the freshly reset game is equivalent and
        # needs no network at all.
        base = self.arc.make(game)
        base.step(GameAction.RESET)          # action_count == 0 => full reset
        self.root = copy.deepcopy(base)
        self.mask = hud_mask(lambda: copy.deepcopy(self.root))
        self.archive: dict[bytes, Entry] = {}
        self.snaps: OrderedDict[bytes, object] = OrderedDict()
        self.snap_cache = snap_cache
        self.best_by_level: dict[int, list[tuple[int, int, int]]] = {}
        self.game_overs = 0
        self.steps = 0        # exploration steps - the search budget
        self.replayed = 0     # steps spent returning to an archived state
        self.max_levels = 0

    # -- state key --------------------------------------------------------
    def key(self, frame: np.ndarray, levels: int) -> bytes:
        masked = np.where(self.mask, np.int8(0), frame)
        return bytes((levels,)) + masked.tobytes()

    # -- restore ----------------------------------------------------------
    def restore(self, e: "Entry"):
        """Return an env sitting at the state `e` describes, and its frame."""
        if e.snap is not None:
            self.snaps.move_to_end(e.key, last=True)
            env = copy.deepcopy(e.snap)
            return env, env._last_response
        env = copy.deepcopy(self.root)
        obs = env._last_response
        for a, x, y in unpack(e.traj):
            act = BY_VALUE[a]
            obs = safe_step(env, act, {"x": x, "y": y} if act.is_complex() else {})
            self.replayed += 1
            if obs is None:
                break
        return env, obs

    def remember(self, e: "Entry", env) -> None:
        """Cache a restorable copy. A snapshot costs 18 ms and 1.2 MiB; replaying
        a depth-d trajectory costs 0.56*d ms, so snapshots pay for themselves
        past roughly 32 steps deep and the LRU keeps the memory bounded."""
        if self.snap_cache <= 0 or e.snap is not None:
            return
        e.snap = copy.deepcopy(env)
        self.snaps[e.key] = e
        self.snaps.move_to_end(e.key, last=True)
        while len(self.snaps) > self.snap_cache:
            _, old = self.snaps.popitem(last=False)
            old.snap = None

    # -- main loop --------------------------------------------------------
    def run(self, budget_steps: int, explore_len: int = 40, verbose: bool = False,
            outdir: Path | None = None, compress_rounds: int = 400) -> dict:
        env = copy.deepcopy(self.root)
        obs = env._last_response
        f = frame_of(obs)
        if f is None:
            return {"game": self.game, "error": "no initial frame"}
        k0 = self.key(f, 0)
        root = Entry(k0, b"", 0)
        self.archive[k0] = root
        self.remember(root, env)

        t0 = time.time()
        next_report = 5000
        while self.steps < budget_steps:
            if verbose and self.steps >= next_report:
                next_report = self.steps + 5000
                el = time.time() - t0
                print(f"    [{self.game}] explore={self.steps} replay={self.replayed} "
                      f"archive={len(self.archive)} snaps={len(self.snaps)} "
                      f"levels={self.max_levels} overs={self.game_overs} {el:.0f}s "
                      f"({self.steps/max(el,1e-9):.0f} explore-steps/s)", flush=True)
            parent = self.select()
            env, obs = self.restore(parent)
            if obs is None:
                obs = env._last_response
            frame = frame_of(obs)
            traj = unpack(parent.traj)
            parent.chosen += 1

            last = None
            for _ in range(explore_len):
                if self.steps >= budget_steps:
                    break
                cands = action_candidates(env, frame, self.rng)
                if not cands:
                    break
                if last is not None and last in cands and self.rng.random() < 0.5:
                    a, x, y = last          # sticky: keep going the same way
                else:
                    a, x, y = self.rng.choice(cands)
                last = (a, x, y)
                act = BY_VALUE[a]
                obs = safe_step(env, act, {"x": x, "y": y} if act.is_complex() else {})
                self.steps += 1
                if obs is None:
                    break          # this action broke the game; abandon the rollout
                traj = traj + [(a, x, y)]

                if obs is None:
                    break
                frame = frame_of(obs)
                levels = int(obs.levels_completed)

                if obs.state is GameState.GAME_OVER:
                    self.game_overs += 1
                    # Only RESET is legal now; it restarts this level and keeps
                    # the levels we already finished.
                    obs = safe_step(env, GameAction.RESET)
                    self.steps += 1
                    if obs is None:
                        break
                    traj = traj + [(GameAction.RESET.value, 0, 0)]
                    frame = frame_of(obs)
                    if frame is None:
                        break
                    levels = int(obs.levels_completed)

                if frame is None:
                    break

                if levels > self.max_levels:
                    self.max_levels = levels
                    print(f"    [{self.game}] level {levels} at {len(traj)} actions "
                          f"({self.steps} sim steps, {time.time() - t0:.0f}s)", flush=True)
                    if outdir is not None:
                        self.best_by_level[levels] = list(traj)
                        (outdir / f"{self.game}.json").write_text(
                            json.dumps(self.report(time.time() - t0), indent=1))
                if levels not in self.best_by_level or len(traj) < len(self.best_by_level[levels]):
                    self.best_by_level[levels] = list(traj)

                k = self.key(frame, levels)
                cur = self.archive.get(k)
                packed = pack(traj)
                if cur is None:
                    e = Entry(k, packed, levels)
                    self.archive[k] = e
                    if len(packed) > 3 * 32:
                        self.remember(e, env)
                elif len(packed) < len(cur.traj):
                    # Same state, cheaper route: keep the cheap one. This is
                    # what turns the archive into an action-count optimiser.
                    cur.traj = packed
                    cur.visits = 0

                if obs.state is GameState.WIN:
                    break

        return self.report(time.time() - t0, compress_rounds=compress_rounds)

    def select(self) -> Entry:
        """Prefer deep progress, then rarely-visited and cheap-to-restore states."""
        best, best_w = None, -1.0
        n = len(self.archive)
        sample = self.rng.sample(list(self.archive.values()), min(n, 64))
        for e in sample:
            w = (e.levels * 1000.0
                 + 1.0 / (1.0 + e.chosen) ** 0.5 * 100.0
                 - len(e.traj) * 0.01
                 + self.rng.random())
            if w > best_w:
                best, best_w = e, w
        best.visits += 1
        return best

    # -- solution compression --------------------------------------------
    def simulate(self, traj: list[tuple[int, int, int]], want_levels: int) -> bool:
        """Does this exact action sequence still finish `want_levels` levels?"""
        env = copy.deepcopy(self.root)
        for a, x, y in traj:
            act = BY_VALUE[a]
            obs = safe_step(env, act, {"x": x, "y": y} if act.is_complex() else {})
            self.replayed += 1
            if obs is None:
                return False
            if int(obs.levels_completed) >= want_levels:
                return True
        return False

    def state_trace(self, traj: list[tuple[int, int, int]]) -> list[bytes]:
        """The archive key after each action, for spotting loops."""
        env = copy.deepcopy(self.root)
        out = []
        for a, x, y in traj:
            act = BY_VALUE[a]
            obs = safe_step(env, act, {"x": x, "y": y} if act.is_complex() else {})
            self.replayed += 1
            if obs is None:
                break
            f = frame_of(obs)
            out.append(self.key(f, int(obs.levels_completed)) if f is not None else b"")
        return out

    def compress(self, traj: list[tuple[int, int, int]], want_levels: int,
                 rounds: int = 400) -> list[tuple[int, int, int]]:
        """Shorten a solution, because the score is quadratic in its length.

        Go-Explore finds A route; the first one it finds wanders. Two passes:
        first remove loops exactly - any stretch that begins and ends in the
        same state can go, since the game is deterministic - then try deleting
        random chunks and keep whatever still wins.
        """
        best = list(traj)

        # Exact loop removal: identical state keys mean an identical situation.
        trace = self.state_trace(best)
        first: dict[bytes, int] = {}
        cut = []
        i = 0
        while i < len(trace):
            k = trace[i]
            if k and k in first:
                cut.append((first[k] + 1, i + 1))
                i += 1
                continue
            first[k] = i
            i += 1
        for lo, hi in reversed(cut):
            cand = best[:lo] + best[hi:]
            if cand and self.simulate(cand, want_levels):
                best = cand

        # Then chip away at what is left.
        for _ in range(rounds):
            if len(best) <= 1:
                break
            span = self.rng.randint(1, max(1, min(12, len(best) // 3)))
            i = self.rng.randrange(0, len(best) - span + 1)
            cand = best[:i] + best[i + span:]
            if cand and self.simulate(cand, want_levels):
                best = cand
        return best

    def report(self, seconds: float, compress_rounds: int = 0) -> dict:
        # The archive holds the cheapest route to every state, including the
        # states that sit just past a level boundary, so it knows shorter
        # solutions than whichever route happened to get there first.
        for e in self.archive.values():
            if e.levels <= 0:
                continue
            cur = self.best_by_level.get(e.levels)
            if cur is None or len(e.traj) // 3 < len(cur):
                self.best_by_level[e.levels] = unpack(e.traj)
        if compress_rounds:
            for L in sorted(self.best_by_level):
                if L <= 0:
                    continue
                before = len(self.best_by_level[L])
                self.best_by_level[L] = self.compress(
                    self.best_by_level[L], L, rounds=compress_rounds)
                print(f"    [{self.game}] level {L}: {before} -> "
                      f"{len(self.best_by_level[L])} actions", flush=True)
        levels = sorted(self.best_by_level)
        per_level: dict[str, int] = {}
        for e in self.archive.values():
            per_level[str(e.levels)] = per_level.get(str(e.levels), 0) + 1
        return {
            "archive_by_level": per_level,
            "game_overs": self.game_overs,
            "game": self.game,
            "sim_steps": self.steps,
            "replay_steps": self.replayed,
            "snapshots": len(self.snaps),
            "seconds": round(seconds, 1),
            "archive": len(self.archive),
            "max_levels": self.max_levels,
            "actions_to_level": {str(L): len(self.best_by_level[L]) for L in levels},
            "solutions": {str(L): self.best_by_level[L] for L in levels},
        }


def solve(game: str, budget: int, seed: int, explore_len: int, verbose: bool,
          outdir: str | None = None) -> dict:
    try:
        s = Solver(game, seed=seed)
        return s.run(budget, explore_len=explore_len, verbose=verbose,
                     outdir=Path(outdir) if outdir else None)
    except Exception:
        import traceback
        return {"game": game, "error": traceback.format_exc(limit=8)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default=None, help="comma list; default = all")
    ap.add_argument("--budget", type=int, default=200_000, help="simulated steps per game")
    ap.add_argument("--explore-len", type=int, default=40)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="solutions")
    args = ap.parse_args()

    quiet()
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    all_ids = [e.game_id.split("-")[0] for e in arc.get_environments()]
    games = [g.strip() for g in args.games.split(",")] if args.games else all_ids

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    def finish(r: dict) -> None:
        if r.get("error"):
            print(f"  {r['game']:6} ERROR {r['error'].splitlines()[-1][:80]}", flush=True)
            return
        base = (outdir / f"{r['game']}.json")
        base.write_text(json.dumps(r, indent=1))
        print(f"  {r['game']:6} levels={r['max_levels']:2}  "
              f"actions={r['actions_to_level']}  archive={r['archive']:6}  "
              f"explore={r['sim_steps']} replay={r['replay_steps']} "
              f"byLevel={r['archive_by_level']} overs={r['game_overs']}  {r['seconds']}s",
              flush=True)

    if args.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = [ex.submit(solve, g, args.budget, args.seed, args.explore_len, False,
                              str(outdir)) for g in games]
            for f in as_completed(futs):
                finish(f.result())
    else:
        for g in games:
            finish(solve(g, args.budget, args.seed, args.explore_len, True, str(outdir)))


if __name__ == "__main__":
    main()
