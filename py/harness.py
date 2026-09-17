r"""Local ARC-AGI-3 evaluation harness with the REAL competition scoring.

Scoring (from arc_agi/scorecard.py + docs.arcprize.org/methodology):
    level_score   = min((baseline_actions / actions_taken)**2, 1.15) if completed else 0
    game_score    = sum(level_score_i * i) / sum(i)        # i = 1-indexed level number
    final_score   = mean(game_score over games)

Run:  .venv\Scripts\python.exe harness.py --agent <module.py> [--games ls20,vc33] [--jobs 10]
"""
from __future__ import annotations

import argparse, importlib.util, json, logging, os, sys, time, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

STARTER = Path(os.environ.get("ARC_STARTER", r"C:\prog\arc\starter"))
sys.path.insert(0, str(STARTER))
sys.path.insert(0, str(STARTER / "vendor" / "ARC-AGI-3-Agents"))


def _load_agent_class(agent_path: Path):
    spec = importlib.util.spec_from_file_location("user_agent_module", agent_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["user_agent_module"] = mod
    spec.loader.exec_module(mod)
    return mod.MyAgent


def _baselines(game_id: str) -> list[int]:
    """Human baseline actions per level, from the downloaded game metadata."""
    short = game_id.split("-")[0]
    root = STARTER / "environment_files" / short
    if not root.exists():
        return []
    for ver in sorted(root.iterdir()):
        meta = ver / "metadata.json"
        if meta.exists():
            return json.loads(meta.read_text())["baseline_actions"]
    return []


def score_game(baselines: list[int], level_actions: list[int], levels_completed: int) -> dict:
    """level_actions[i] = actions spent on level i (completed ones first)."""
    total_w = sum(range(1, len(baselines) + 1))
    got = 0.0
    per_level = []
    for i, base in enumerate(baselines):
        w = i + 1
        if i < levels_completed and i < len(level_actions) and level_actions[i] > 0:
            s = min((base / level_actions[i]) ** 2, 1.15)
        else:
            s = 0.0
        per_level.append(round(s, 4))
        got += s * w
    return {"game_score": 100.0 * got / total_w if total_w else 0.0, "per_level": per_level}


def play_one(agent_path: str, game_id: str, max_actions: int, render: str | None) -> dict:
    import arc_agi
    from arc_agi import OperationMode
    logging.getLogger().setLevel(logging.ERROR)
    for n in ("arc_agi", "arc_agi.scorecard", "root"):
        logging.getLogger(n).setLevel(logging.ERROR)

    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    env = arc.make(game_id.split("-")[0], render_mode=render)
    if env is None:
        return {"game_id": game_id, "error": "no env"}

    # Read the human baselines only after make(), which is what downloads the
    # game (and its metadata.json) on first use.
    baselines = _baselines(game_id)
    budget = min(sum(5 * b for b in baselines) if baselines else max_actions, max_actions)

    AgentCls = _load_agent_class(Path(agent_path))
    AgentCls.MAX_ACTIONS = budget
    agent = AgentCls(card_id="local", game_id=game_id, agent_name=f"h.{game_id}",
                     ROOT_URL="http://localhost", record=False, arc_env=env, tags=["harness"])

    # Track per-level action spend by watching levels_completed transitions.
    level_actions: list[int] = []
    last_lc, last_count = 0, 0
    orig_append = agent.append_frame

    def append_frame(frame):
        nonlocal last_lc, last_count
        orig_append(frame)
        if frame.levels_completed > last_lc:
            for _ in range(frame.levels_completed - last_lc):
                level_actions.append(agent.action_counter - last_count)
                last_count = agent.action_counter
            last_lc = frame.levels_completed
    agent.append_frame = append_frame

    t0 = time.time()
    err = None
    try:
        agent.main()
    except Exception:
        err = traceback.format_exc(limit=6)
    dt = time.time() - t0

    final = agent.frames[-1]
    res = score_game(baselines, level_actions, final.levels_completed)
    return {"game_id": game_id.split("-")[0], "levels_completed": final.levels_completed,
            "n_levels": len(baselines), "actions": agent.action_counter, "budget": budget,
            "level_actions": level_actions, "baselines": baselines, "seconds": round(dt, 1),
            "state": str(final.state), "error": err, **res}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True)
    ap.add_argument("--games", default=None, help="comma list; default = all")
    ap.add_argument("--max-actions", type=int, default=100_000)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--render", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import arc_agi
    from arc_agi import OperationMode
    logging.getLogger().setLevel(logging.ERROR)
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    all_ids = [e.game_id.split("-")[0] for e in arc.get_environments()]
    games = [g.strip() for g in args.games.split(",")] if args.games else all_ids

    agent_path = str(Path(args.agent).resolve())
    results = []
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(play_one, agent_path, g, args.max_actions, None): g for g in games}
            for f in as_completed(futs):
                r = f.result(); results.append(r)
                print(f"  {r['game_id']:6} lv={r.get('levels_completed',0)}/{r.get('n_levels','?')} "
                      f"act={r.get('actions',0):6} score={r.get('game_score',0):6.2f}  ({r.get('seconds',0)}s)",
                      flush=True)
    else:
        for g in games:
            r = play_one(agent_path, g, args.max_actions, args.render); results.append(r)
            print(f"  {r['game_id']:6} lv={r.get('levels_completed',0)}/{r.get('n_levels','?')} "
                  f"act={r.get('actions',0):6} score={r.get('game_score',0):6.2f}  ({r.get('seconds',0)}s)",
                  flush=True)

    results.sort(key=lambda r: r["game_id"])
    final = sum(r.get("game_score", 0.0) for r in results) / max(len(results), 1)
    print("\n================ RESULT ================")
    for r in results:
        if r.get("error"):
            print(f"  {r['game_id']:6} ERROR: {r['error'].splitlines()[-1][:90]}")
    print(f"  games={len(results)}  levels={sum(r.get('levels_completed',0) for r in results)}"
          f"  FINAL SCORE = {final:.2f} / 100")
    if args.out:
        Path(args.out).write_text(json.dumps({"final": final, "results": results}, indent=1))


if __name__ == "__main__":
    main()
