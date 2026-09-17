r"""Play one game with the C++ core and report what it believes.

Prints the core's internal state (avatar found? where? which action moves it?)
so we can see whether a zero score is a perception failure or a planning one.
"""
from __future__ import annotations

import argparse, ctypes, importlib.util, logging, os, sys
from collections import Counter
from pathlib import Path

STARTER = Path(os.environ.get("ARC_STARTER", r"C:\prog\arc\starter"))
sys.path.insert(0, str(STARTER))
sys.path.insert(0, str(STARTER / "vendor" / "ARC-AGI-3-Agents"))

import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction, GameState

ap = argparse.ArgumentParser()
ap.add_argument("--game", default="ls20")
ap.add_argument("--steps", type=int, default=400)
ap.add_argument("--agent", default=str(Path(__file__).resolve().parents[1] / "agent" / "my_agent.py"))
ap.add_argument("--every", type=int, default=25)
ap.add_argument("--show-frame", action="store_true")
args = ap.parse_args()

logging.getLogger().setLevel(logging.ERROR)
for n in ("arc_agi", "arc_agi.scorecard", "root"):
    logging.getLogger(n).setLevel(logging.ERROR)

spec = importlib.util.spec_from_file_location("ua", args.agent)
mod = importlib.util.module_from_spec(spec)
sys.modules["ua"] = mod
spec.loader.exec_module(mod)

core = mod._core()
print(f"C++ core loaded: {core is not None}")

arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
env = arc.make(args.game)
AgentCls = mod.MyAgent
AgentCls.MAX_ACTIONS = args.steps
agent = AgentCls(card_id="dbg", game_id=args.game, agent_name="dbg",
                 ROOT_URL="http://localhost", record=False, arc_env=env, tags=["dbg"])

stats = (ctypes.c_int * 32)()
action_hist = Counter()
step = 0


def dump(tag: str) -> None:
    if core is None or agent._h < 0:
        return
    core.arc3_stats(agent._h, stats)
    names = ["known", "color", "ax", "ay", "steps", "levels", "trig", "blocked",
             "w", "h", "plan", "stag", "touched", "esc", "states", "restarts",
             "rules", "goals", "walls"]
    print(f"  [{tag}] " + "  ".join(f"{n}={stats[i]}" for i, n in enumerate(names)))


orig_take = agent.take_action


def take_action(action):
    global step
    step += 1
    action_hist[action.name] += 1
    f = orig_take(action)
    if step % args.every == 0:
        lc = f.levels_completed if f else "?"
        print(f"step {step:5}  action={action.name:8} levels={lc}")
        dump("core")
    return f


agent.take_action = take_action
agent.main()

final = agent.frames[-1]
print(f"\nfinal: state={final.state} levels_completed={final.levels_completed} actions={agent.action_counter}")
dump("final")
print("action histogram:", dict(action_hist.most_common()))

if args.show_frame and final.frame:
    g = final.frame[-1]
    palette = " .:-=+*#%@ABCDEF"
    print("\nfinal frame:")
    for row in g:
        print("".join(palette[v % 16] for v in row))
