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

press, click and restart each cost one action and give the board back changed.
Your score is (the number of actions a human needed / the number you used),
squared. Finishing a level in 30 actions where a human took 25 scores well;
finishing it in 500 scores nothing at all. So look before you act, and do not
sweep the board.
"""

PREAMBLE = """\
Write a Lua script that finishes this level. Reply with Lua only: no markdown
fence, no explanation, no comments about what you cannot do.

Work it out from the board and from what the actions do. Use say() to record
what you concluded, so that if the script fails I can tell you why.
"""


def ask(prompt: str, n_predict: int = 700) -> str:
    """One call to the model. Thinking is off; it answers with the script."""
    p = Path(os.environ.get("TEMP", ".")) / "lua_llm_prompt.txt"
    p.write_text(prompt, encoding="utf-8")
    r = subprocess.run(
        [str(CLI), "-m", str(MODEL), "-f", str(p), "-st", "-ngl", "99",
         "-c", "16384", "-n", str(n_predict), "--temp", "0.2",
         "--jinja", "--reasoning-budget", "0", "--no-display-prompt"],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = r.stdout or ""
    # llama-cli writes its banner and timings around the answer
    lines = [l for l in out.splitlines()
             if all(ord(c) < 128 for c in l)
             and not l.startswith(("build ", "model ", "ftype", "modalities",
                                   "[ Prompt", "Exiting", "load", "llama_"))]
    text = "\n".join(lines)
    text = re.sub(r"^```[a-z]*$|^```$", "", text, flags=re.M)
    return text.strip()


def describe(game: str) -> str:
    """The board and what each action does, in the vocabulary of objects."""
    from llm_probe import probe
    return probe(game)


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
        prompt = f"{API}\n{board}\n{history}\n{PREAMBLE}"
        script = ask(prompt)
        if not script:
            print(f"round {rnd}: the model said nothing")
            continue
        r = lua_agent.play(lib, args.game, script, args.max, False)
        log = (r.get("log") or r.get("error") or "").strip()
        print(f"--- round {rnd}: levels={r.get('levels', 0)} actions={r.get('actions', 0)}")
        if log:
            print(textwrap.indent(log[:1200], "    "))
        if args.keep:
            Path(args.keep).with_suffix(f".r{rnd}.lua").write_text(script, encoding="utf-8")
        if best is None or r.get("levels", 0) > best.get("levels", 0):
            best = dict(r, script=script)
        if r.get("levels", 0) > 0:
            break
        history = ("\nYour previous script ran and this is what happened. "
                   f"It finished {r.get('levels', 0)} levels in {r.get('actions', 0)} "
                   "actions. Its log was:\n" + (log[:1500] or "(nothing)") +
                   "\nWrite a better script.\n")

    print(f"\nBEST game={args.game} levels={best.get('levels', 0) if best else 0} "
          f"actions={best.get('actions', 0) if best else 0}")


if __name__ == "__main__":
    main()
