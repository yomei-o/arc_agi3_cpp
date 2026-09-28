r"""Ask for the answer, not for a search.

Every prompt before this one told the model to press a button, look at what
changed, and decide - that is, to write a search. Search is what the scoring
punishes: the score is (the actions a human needed / the actions we spent),
squared, and a human finishes lp85's first level in seventeen actions. Five
hundred actions of careful exploration scores zero, and so does five hundred
actions of anything else.

So this asks for a plan: a literal list of moves, no loops, no conditions. If
it is wrong we show what the board became and ask again. Three wrong plans of
twenty moves still cost sixty actions, which is a tenth of what searching cost.

    python plan_play.py --game lp85 --tries 6 --moves 30
"""
from __future__ import annotations

import argparse, ctypes, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lua_agent
from lua_llm import ask

ASK = """\
Here is one level of a grid puzzle video game, described as objects on a grid of
cells. Coordinates count from 0.

{board}
{story}
Write the moves that finish this level. One move per line, nothing else:

  press 1      press a button, 1 to {maxbutton}
  click 12 7   click the cell at x=12, y=7

No loops, no conditions, no comments, no explanation - just the moves. At most
{moves} of them.

You are not told what the buttons do. If you have been told below what they did,
use it; otherwise guess, and you will be told what happened.

Your score is (the moves a human needed / the moves you spend), squared, counted
over every attempt at this level. A human finishes a level like this in a few
tens of moves. So write the shortest plan you believe in, not a careful search.
"""

MOVE = re.compile(r"^\s*(?:press\s+([1-5])|click\s+(\d+)\s+(\d+))\s*$", re.I)


def to_lua(text: str, cap: int) -> tuple:
    """Turn the model's lines into a Lua script that just does them."""
    steps = []
    for line in text.splitlines():
        m = MOVE.match(line)
        if not m:
            continue
        if m.group(1):
            steps.append("press(%s)" % m.group(1))
        else:
            steps.append("click(%s, %s)" % (m.group(2), m.group(3)))
        if len(steps) >= cap:
            break
    return "\n".join(steps), len(steps)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="lp85", help="one id, a comma list, or 'all'")
    ap.add_argument("--tries", type=int, default=6)
    ap.add_argument("--moves", type=int, default=30)
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()

    lib = lua_agent.build()
    if args.game == "all":
        lua_agent.quiet()
        arc = lua_agent.arc_agi.Arcade(operation_mode=lua_agent.OperationMode.NORMAL)
        games = sorted({e.game_id.split("-")[0] for e in arc.get_environments()})
    else:
        games = [g.strip() for g in args.game.split(",")]

    if len(games) > 1 and args.jobs > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            rows = list(pool.map(lambda g: one(lib, g, args), games))
    else:
        rows = [one(lib, g, args) for g in games]

    won = [r for r in rows if r[1] > 0]
    print("\n%d/%d ゲームで1レベル以上: %s"
          % (len(won), len(rows),
             ", ".join("%s(%dlv, 到達%s手)" % (a, b, c) for a, b, c, _ in won)), flush=True)


def one(lib, game: str, args) -> tuple:
    sess = lua_agent.Session(lib, game)
    buf = ctypes.create_string_buffer(65536)
    lib.arc3_lua_state.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    lib.arc3_lua_state.restype = ctypes.c_int

    maxbutton = max([a for a in sess.avail if 1 <= a <= 5] or [1])
    story = ""
    for attempt in range(1, args.tries + 1):
        lib.arc3_lua_state(sess.h, buf, len(buf))
        board = buf.value.decode("utf-8", "replace")
        plan_text = ask(ASK.format(board=board, story=story, moves=args.moves,
                                   maxbutton=maxbutton), n_predict=400, raw=True)
        script, n = to_lua(plan_text, args.moves)
        if n == 0:
            print("  %s attempt %d: no moves parsed" % (game, attempt), flush=True)
            story = "\nYour last reply had no moves in it. Write only lines like 'press 3'.\n"
            continue

        # Start each attempt from the top of the level, the way a person
        # retries. Otherwise the model plans from a board that its own half-
        # finished previous plan left behind, and cannot reason about either.
        if attempt > 1:
            sess.run("restart()", 2)

        before = sess.levels
        r = sess.run(script, args.moves + 2)
        gained = sess.levels - before
        print("  %s attempt %d: %d moves, spent %d, level=%d%s"
              % (game, attempt, n, r["spent"], sess.levels, "  LEVEL UP" if gained else ""),
              flush=True)
        if gained or sess.done:
            break
        lib.arc3_lua_state(sess.h, buf, len(buf))
        story = ("\nYou tried this plan and it did not finish the level:\n"
                 + script + "\nAfterwards the board was:\n"
                 + buf.value.decode("utf-8", "replace")
                 + "\nWrite a different plan.\n")

    print("\nRESULT game=%s levels=%d actions=%d level_actions=%s"
          % (game, sess.levels, sess.used, sess.level_actions), flush=True)
    out = (game, sess.levels, sess.level_actions, sess.used)
    sess.close()
    return out


if __name__ == "__main__":
    main()
