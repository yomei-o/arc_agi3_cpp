"""Splice cpp/arc3_core.cpp into py/agent_template.py -> agent/my_agent.py.

The Kaggle notebook builder ships exactly one file, so the C++ core travels
inside the agent as a string and is compiled on the Kaggle machine at startup.
Keeping the C++ in its own file means it still compiles and gets -Wall checked
during development; this script is the only thing that joins them.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CPP = ROOT / "cpp" / "arc3_core.cpp"
TPL = ROOT / "py" / "agent_template.py"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "agent" / "my_agent.py"

cpp = CPP.read_text(encoding="utf-8")
# The C++ goes inside r"""...""", so it must not contain that delimiter or end
# with a backslash. Neither is true today; assert so it stays that way.
if '"""' in cpp:
    raise SystemExit('arc3_core.cpp contains a triple quote; cannot embed verbatim')
if cpp.rstrip().endswith("\\"):
    raise SystemExit("arc3_core.cpp ends with a backslash; cannot embed verbatim")

tpl = TPL.read_text(encoding="utf-8")
if "__ARC3_CPP_SOURCE__" not in tpl:
    raise SystemExit("template lost its __ARC3_CPP_SOURCE__ placeholder")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(tpl.replace("__ARC3_CPP_SOURCE__", cpp), encoding="utf-8")
print(f"[bundle] {OUT}  ({len(cpp.splitlines())} lines of C++ embedded)")
