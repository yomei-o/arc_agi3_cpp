r"""Grade the LLM's answers against what the simulator actually showed it.

Only the first question can be graded mechanically. "What does the player
control" has an answer the prompt itself contains: the object the simulator
reports moving when a movement button is pressed. So this checks whether the
model can read its own evidence - a low bar, but a real one, and the 8B failed
it on five games out of six.

The second question - what the level wants - has no key here. Deciding that
needs the level played, which is the next experiment, not this one.

    python grade_llm.py --prompts C:\models\prompts --answers C:\models\a27b.txt
"""
from __future__ import annotations

import argparse, re
from pathlib import Path

MOVED = re.compile(r"the colour-(\S) object of (\d+) cells moved \([-+\d]+,[-+\d]+\) "
                   r"from \((\d+),(\d+)\) to \((\d+),(\d+)\)")


def truth(prompt: str):
    """The object the sim shows moving under a movement button, if any."""
    for line in prompt.splitlines():
        if not line.startswith("  ACTION"):
            continue
        m = MOVED.search(line)
        if m:
            g, _, x, y, _, _ = m.groups()
            return g, int(x), int(y)
    return None


def answers(path: Path) -> dict:
    out, cur, buf = {}, None, []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("#####"):
            if cur:
                out[cur] = "\n".join(buf)
            cur, buf = line.split()[1], []
        elif cur and not line.startswith("["):
            buf.append(line)
    if cur:
        out[cur] = "\n".join(buf)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--answers", required=True)
    args = ap.parse_args()

    ans = answers(Path(args.answers))
    hit = miss = skip = 0
    for f in sorted(Path(args.prompts).glob("*.txt")):
        g = f.stem
        t = truth(f.read_text(encoding="utf-8"))
        a = ans.get(g, "")
        if t is None:
            print(f"  {g}: 動く物体が無い(クリック専用など) - 採点対象外")
            skip += 1
            continue
        glyph, x, y = t
        said_coord = f"({x},{y})" in a or f"({x}, {y})" in a
        said_glyph = f"'{glyph}'" in a or f"colour-{glyph}" in a or f" {glyph} object" in a
        ok = said_coord or said_glyph
        hit += ok
        miss += not ok
        mark = "OK " if ok else "NG "
        print(f"  {mark}{g}: 正解は '{glyph}' at ({x},{y})"
              f"{'' if ok else '  -> ' + ' '.join(a.split())[:90]}")
    total = hit + miss
    print(f"\n操作対象を当てた: {hit}/{total}"
          f"（採点対象外 {skip}）" if total else "\n採点対象なし")


if __name__ == "__main__":
    main()
