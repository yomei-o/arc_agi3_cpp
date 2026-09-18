"""Splice the C++ core into py/agent_template.py -> agent/my_agent.py.

The Kaggle notebook builder ships exactly one file, so the C++ travels inside
the agent as a string and is compiled on the Kaggle machine at startup. The C++
is spread over several files during development - the agent core, the network,
and the vendored autograd engine - so local `#include "..."` directives are
expanded here into one translation unit. System includes are left alone, and a
header included more than once is emitted once.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CPP = ROOT / "cpp" / "arc3_core.cpp"
TPL = ROOT / "py" / "agent_template.py"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "agent" / "my_agent.py"

_LOCAL_INCLUDE = re.compile(r'^\s*#\s*include\s+"([^"]+)"\s*$')


def expand(path: Path, seen: set[Path]) -> list[str]:
    path = path.resolve()
    if path in seen:
        return []
    seen.add(path)
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = _LOCAL_INCLUDE.match(line)
        if m:
            target = (path.parent / m.group(1)).resolve()
            if target.exists():
                out.append(f"// ---- begin {target.name} ----")
                out += expand(target, seen)
                out.append(f"// ---- end {target.name} ----")
                continue
        out.append(line)
    return out


seen: set[Path] = set()
lines = expand(CPP, seen)

# The autograd engine lives in a .cpp beside its header; pull it in too, since a
# single translation unit is the whole point.
impl = ROOT / "cpp" / "ag" / "autograd.cpp"
if impl.resolve() not in seen and impl.exists():
    lines.append(f"// ---- begin {impl.name} ----")
    lines += expand(impl, seen)
    lines.append(f"// ---- end {impl.name} ----")

cpp = "\n".join(lines) + "\n"

# The C++ goes inside r"""...""", so it must not contain that delimiter or end
# with a backslash. Neither is true today; assert so it stays that way.
if '"""' in cpp:
    raise SystemExit("the C++ sources contain a triple quote; cannot embed verbatim")
if cpp.rstrip().endswith("\\"):
    raise SystemExit("the C++ sources end with a backslash; cannot embed verbatim")

tpl = TPL.read_text(encoding="utf-8")
if "__ARC3_CPP_SOURCE__" not in tpl:
    raise SystemExit("template lost its __ARC3_CPP_SOURCE__ placeholder")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(tpl.replace("__ARC3_CPP_SOURCE__", cpp), encoding="utf-8")
print(f"[bundle] {OUT}  ({len(cpp.splitlines())} lines of C++ from {len(seen)} files)")
