r"""Play a game by asking the model for the next few moves, over and over.

One script written at the start cannot play these games: the model has not seen
the level yet, and the thing it needs to know - what the buttons do, what the
goal reacts to - is only learned by acting. So the game, and everything the host
has learned about it, outlive each script. The model writes a few moves, sees
what they did, and writes a few more.

    python lua_play.py --game ka59 --turns 8 --budget 120

Runs on the build machine: it needs the ARC environment, g++, and the model.
"""
from __future__ import annotations

import argparse, ctypes, os, sys, textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lua_agent
from lua_llm import ask

API = """\
You are playing one level of a grid puzzle game by writing short Lua scripts.
You write a few moves, I run them, and I show you what happened. Then you write
the next few. The game does not restart between your scripts and everything you
have learned stays learned.

These functions exist and NOTHING else does. There is no io, no os, no require.

  objects()      -> array of objects: x, y, colour, glyph, area, w, h, minx, miny
  changes()      -> what the LAST action did: kind = "moved" (with dx, dy, fx, fy),
                    "appeared" or "vanished", each with colour, area, x, y
  at(x, y)       -> the colour of that cell, or nil off the board
  size()         -> width, height in cells
  level()        -> how many levels are finished
  steps()        -> how many actions used so far
  actions()      -> which buttons this game offers
  me()           -> the object you control, or nil until something has moved
  walls()        -> the cells that have been found impassable
  say(text)      -> write a line I will show you afterwards

  press(n)       -> press button n (1..5). Returns true if the board changed.
  click(x, y)    -> click that cell. Returns true if the board changed.
  move_to(x, y)  -> walk there by the shortest route known, avoiding walls.
                    Returns true on arrival, false if there is no route.
                    It costs one action per step and is the cheapest way to get
                    anywhere. Do not write your own walking loop.
  explore(n)     -> hand the next n actions to a search that is good at finding
                    what changes a board when you do not yet know what to try.
                    Use it when you have no hypothesis; stop using it when you do.
  restart()      -> restart the current level.

Coordinates are cells, counted from 0. objects() is the truth about positions.

Your score is (the actions a human needed / the actions you used), squared. A
level finished in 30 actions where a human took 25 scores well; the same level
in 500 scores nothing. Spend actions on a hypothesis, not on sweeping.

But you get only a handful of turns, so a script that spends five actions and
hands back is worse than useless - it burns a turn to learn almost nothing.
Write a script that keeps going: loop, test a hypothesis, and if it fails, try
the next one within the same script. When you have no hypothesis at all, call
explore(200) and look at what moved. Your script should normally spend most of
the actions it is allowed.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="ka59", help="one id, a comma list, or 'all'")
    ap.add_argument("--turns", type=int, default=8)
    ap.add_argument("--budget", type=int, default=120, help="actions per turn")
    ap.add_argument("--keep", default="")
    ap.add_argument("--jobs", type=int, default=4, help="games to play at once")
    args = ap.parse_args()

    lib = lua_agent.build()
    if args.game == "all":
        lua_agent.quiet()
        arc = lua_agent.arc_agi.Arcade(operation_mode=lua_agent.OperationMode.NORMAL)
        games = sorted({e.game_id.split("-")[0] for e in arc.get_environments()})
    else:
        games = [g.strip() for g in args.game.split(",")]

    # One game tells you nothing; every wrong remedy this month looked right on
    # the game it was found on.
    #
    # The games are independent, and the resident model batches concurrent
    # requests, so running them together costs almost nothing and turns an hour
    # into minutes. A day spent changing one prompt at a time is a day of
    # waiting if each change takes an hour to judge.
    if len(games) > 1 and args.jobs > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            summary = list(pool.map(lambda g: play_one(lib, g, args), games))
    else:
        summary = [play_one(lib, g, args) for g in games]
    if len(summary) > 1:
        won = [r for r in summary if r[1] > 0]
        print("\n%d/%d ゲームで1レベル以上: %s"
              % (len(won), len(summary),
                 ", ".join("%s(%dlv/%d手)" % (a, b, c) for a, b, c in won)), flush=True)


def play_one(lib, game: str, args) -> tuple:
    sess = lua_agent.Session(lib, game)

    buf = ctypes.create_string_buffer(65536)
    lib.arc3_lua_state.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    lib.arc3_lua_state.restype = ctypes.c_int

    def briefing() -> str:
        lib.arc3_lua_state(sess.h, buf, len(buf))
        return buf.value.decode("utf-8", "replace")

    story, notes = "", []
    for turn in range(1, args.turns + 1):
        prompt = (API + "\nWhat the board looks like now:\n" + briefing() + story +
                  f"\nYou have {args.turns - turn + 1} turns left and may spend up"
                  f" to {args.budget} actions in this one. Write Lua only: no"
                  " fence, no prose, under forty lines - a script cut off in the"
                  " middle does not run at all.\n")
        # Forty lines of Lua is about four hundred tokens. Asking for 1200
        # does not make the script better, it makes every turn three times
        # longer: the model fills whatever room it is given, and at twenty
        # tokens a second on a shared card that is a minute a turn.
        script = ask(prompt, n_predict=900)
        if args.keep:
            Path(args.keep + f".t{turn}.lua").write_text(script, encoding="utf-8")
        if not script.strip():
            print(f"turn {turn}: the model wrote nothing", flush=True)
            break

        before_lv, before_st = sess.levels, sess.used
        r = sess.run(script, args.budget)

        # A syntax error is worth one more call, not a whole turn. The model
        # fixes its own Lua when shown the message, and a call is now seconds -
        # where a turn is the scarce thing, because the game only moves on a
        # script that runs.
        if not r["loaded"]:
            fix = ask("%s\nYou wrote this and it did not compile:\n%s\n"
                      "The error was: %s\nWrite it again, correctly."
                      % (prompt, script[:1500], r["log"][:300]), n_predict=900)
            if fix.strip():
                script = fix
                r = sess.run(script, args.budget)
        gained = sess.levels - before_lv
        print(f"--- turn {turn}: spent={r['spent']} total={sess.used} "
              f"level={sess.levels}"
              + ("  LEVEL UP" if gained else "")
              + ("" if r["loaded"] else "  DID NOT COMPILE"), flush=True)
        if r["log"]:
            print(textwrap.indent(r["log"][:600], "    "), flush=True)

        if sess.done:
            print("game won", flush=True)
            break
        # Keep a few turns of what it concluded, not just the last one.
        # With one turn of memory it rediscovered the same wall three times and
        # spent a third of the game doing it.
        note = ("\nTurn %d: spent %d actions%s%s. It said:\n%s\n"
                % (turn, r["spent"],
                   ", finished a level" if gained else "",
                   "" if r["loaded"] else ", DID NOT COMPILE - write valid Lua",
                   r["log"][:600] or "(nothing)"))
        notes.append(note)
        story = "".join(notes[-3:])

    print("\nRESULT game=%s levels=%d actions=%d level_actions=%s"
          % (game, sess.levels, sess.used, sess.level_actions), flush=True)
    lv, used = sess.levels, sess.used
    sess.close()
    return (game, lv, used)


if __name__ == "__main__":
    main()
