r"""Let the model write the policy, run it, and tell it what happened.

This is the loop the leading harness uses, with one difference that is the whole
bet: it hands the model raw actions and lets it write the search; we hand it
primitives that already know what an object is and how to see what an action
did. The score is the ratio of human actions to ours, squared, so the cost of a
clumsy inner loop is paid at compound interest - and a loop written in C++ is
not clumsy.

    python lua_llm.py --game ka59 --rounds 3 --max 1500

Runs on the build machine: it needs the ARC environment, g++, and the model.
"""
from __future__ import annotations

import argparse, os, re, subprocess, sys, textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

CLI = Path(os.environ.get("LLAMA_CLI", r"C:\prog\llama.cpp\build-cuda\bin\Release\llama-cli.exe"))
MODEL = Path(os.environ.get("LLAMA_MODEL", r"C:\models\Qwen3.6-27B-Q4_K_M.gguf"))

GLYPHS = ".:-=+*#%@ABCDEFG"

API = """\
You are writing a Lua policy that plays one level of a grid puzzle game.
The script runs to completion; there is no main loop to return to.

These functions exist and NOTHING else does. There is no io, no os, no require.

  objects()      -> array of objects on the board, each a table with
                    x, y      the centre, in cells
                    colour    an integer
                    glyph     the character used for it in the board text
                    area      size in cells
                    w, h      bounding box, in cells
                    minx,miny top-left corner, in cells
  changes()      -> what the LAST action did, as an array of events:
                    kind = "moved"     with colour, area, fx, fy, x, y, dx, dy
                    kind = "appeared"  with colour, area, x, y
                    kind = "vanished"  with colour, area, x, y
  at(x, y)       -> the colour of that cell, or nil off the board
  size()         -> width, height in cells
  board_text()   -> the whole board as text, with coordinate rulers
  level()        -> how many levels are finished
  steps()        -> how many actions used so far
  actions()      -> which buttons this game offers
  say(text)      -> write a line to the log; this is how you tell me what you saw

  press(n)       -> press button n (1..5). Returns true if the board changed.
  click(x, y)    -> click that cell. Returns true if the board changed.
  restart()      -> restart the current level.

  me()           -> the object you control, or nil until something has moved.
                    Press each button once at the start and it will know.
  walls()        -> the cells found to be impassable so far.
  move_to(x, y)  -> walk there by the shortest route known, avoiding walls.
                    Returns true if it arrived, false if there is no route.
                    This costs one action per step, and it is the cheapest way
                    to get anywhere: do not write your own walking loop.

Coordinates are cells, counted from 0. objects() is the truth about what is on
the board; do not work positions out any other way.

press, click and restart each cost one action and give the board back changed.
Your score is (the number of actions a human needed / the number you used),
squared. Finishing a level in 30 actions where a human took 25 scores well;
finishing it in 500 scores nothing at all. So look before you act, and do not
sweep the board.
"""

PREAMBLE = """\
Write a Lua script that finishes this level. Reply with Lua only: no markdown
fence, no prose, no apology for what you cannot see.

You are not told the rules; work them out. A good script presses a button,
looks at changes(), and decides what to do next - it does not assume. Use say()
to record what you concluded, so that if it fails I can tell you why.

Keep it under sixty lines. A script that is cut off in the middle does not run
at all, and three of your last three were.
"""


def ask(prompt: str, n_predict: int = 1600) -> str:
    """One call to the model. Thinking is off; it answers with the script.

    The answer is found by subtracting the prompt, not by hunting for where it
    starts. llama-cli echoes what it was given and prints a banner, and the
    banner goes to stderr so it interleaves at no fixed place. Position cannot
    separate them; set membership can. The first filter let "Loading model..."
    through and Lua was handed it as a program.
    """
    p = Path(os.environ.get("TEMP", ".")) / "lua_llm_prompt.txt"
    p.write_text(prompt, encoding="utf-8")
    r = subprocess.run(
        [str(CLI), "-m", str(MODEL), "-f", str(p), "-st", "-ngl", "99",
         "-c", "16384", "-n", str(n_predict), "--temp", "0.2",
         "--jinja", "--reasoning-budget", "0", "--no-display-prompt"],
        capture_output=True, text=True, encoding="utf-8", errors="replace")

    noise = ("build ", "model ", "ftype", "modalities", "[ Prompt", "Exiting",
             "load", "llama_", "Loading", "available commands", "/exit",
             "/regen", "/clear", "/read", "/glob", ">")
    plines = [l.strip() for l in prompt.splitlines() if l.strip()]
    echoed = set(plines)
    keep = []
    for line in (r.stdout or "").splitlines():
        t = line.strip()
        if not all(ord(c) < 128 for c in line):
            continue
        if t in echoed or t.startswith(noise) or "(truncated)" in t:
            continue
        # llama-cli cuts the echoed prompt mid-line, so the tail of the cut line
        # matches nothing in the set. A prefix of a prompt line is still prompt.
        if t and any(q.startswith(t) for q in plines):
            continue
        keep.append(line)
    text = "\n".join(keep)
    text = re.sub(r"^```[a-z]*$|^```$", "", text, flags=re.M)

    # Drop whatever prose came before the model started coding. Lua has no
    # statement that begins with a digit or an article, so a line that does is
    # not part of the program.
    start = re.compile(r"^\s*(--|local\b|for\b|while\b|if\b|function\b|repeat\b"
                       r"|do\b|return\b|[A-Za-z_][A-Za-z_0-9]*\s*[=(])")
    out = text.splitlines()
    while out and not start.match(out[0]):
        out.pop(0)
    return "\n".join(out).strip()


def describe(game: str) -> str:
    """What is on the board and what each action does - as objects, not pixels.

    The first version pasted the board as text as well. The model then ignored
    the object list it had been given, counted columns in the ASCII by hand, got
    a different answer, argued with itself for four hundred tokens and emitted
    no program at all. Which is the day's lesson arriving from the other side: a
    model reasons in the vocabulary it is given, so handing it two vocabularies
    and letting it choose is worse than handing it one.
    """
    from llm_probe import probe
    out, skipping = [], False
    for line in probe(game).splitlines():
        # probe() ends by asking two questions of its own. Leaving them in meant
        # the prompt carried two requests, and the model answered the one it was
        # given first: three rounds of "1. The player controls..." and not a
        # line of Lua. One prompt, one request.
        if line.startswith("Answer in at most"):
            break
        if line.startswith("BOARD:"):
            skipping = True
            continue
        if skipping:
            if line.strip() and all(c in GLYPHS for c in line.strip()):
                continue
            skipping = False
        out.append(line)
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="ka59")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--max", type=int, default=1500)
    ap.add_argument("--keep", default="")
    args = ap.parse_args()

    import lua_agent
    lib = lua_agent.build()

    board = describe(args.game)
    history = ""
    best = None
    for rnd in range(1, args.rounds + 1):
        prompt = API + "\n" + board + "\n" + history + "\n" + PREAMBLE
        script = ask(prompt)
        if not script:
            print(f"round {rnd}: the model said nothing", flush=True)
            continue
        if args.keep:
            Path(args.keep + f".r{rnd}.lua").write_text(script, encoding="utf-8")
        r = lua_agent.play(lib, args.game, script, args.max, False)
        log = (r.get("log") or r.get("error") or "").strip()
        print(f"--- round {rnd}: levels={r.get('levels', 0)} "
              f"actions={r.get('actions', 0)}", flush=True)
        if log:
            print(textwrap.indent(log[:1200], "    "), flush=True)
        if best is None or r.get("levels", 0) > best.get("levels", 0):
            best = dict(r, script=script)
        if r.get("levels", 0) > 0:
            break
        history = ("\nYour previous script ran. It finished "
                   f"{r.get('levels', 0)} levels in {r.get('actions', 0)} actions. "
                   "Its log was:\n" + (log[:1500] or "(nothing)") +
                   "\nWrite a better script.\n")

    lv = best.get("levels", 0) if best else 0
    ac = best.get("actions", 0) if best else 0
    print(f"\nBEST game={args.game} levels={lv} actions={ac}", flush=True)


if __name__ == "__main__":
    main()
