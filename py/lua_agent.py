r"""Play a real ARC game with a policy written in Lua.

The C++ side holds the board, the object extractor and the Lua state; this file
only owns the environment. The loop is the one the coroutine design implies:
show the board, resume the script, take whatever action it asks for, step the
game, repeat. The script is never told it was suspended.

    python lua_agent.py --game ka59 --script ../lua/walk.lua --max 2000
"""
from __future__ import annotations

import argparse, ctypes, os, subprocess, sys, tempfile
from pathlib import Path

STARTER = Path(os.environ.get("ARC_STARTER", r"C:\prog\arc\starter"))
sys.path.insert(0, str(STARTER))
sys.path.insert(0, str(STARTER / "vendor" / "ARC-AGI-3-Agents"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction

from solve_offline import BY_VALUE, frame_of, quiet

ROOT = Path(__file__).resolve().parent.parent
LUA_SRC = Path(os.environ.get("LUA_SRC", r"C:\prog\lua-5.4.8\src"))

_STATE_CODE = {"NOT_PLAYED": 0, "NOT_FINISHED": 1, "WIN": 2, "GAME_OVER": 3}


def build() -> ctypes.CDLL:
    """Compile the core with the Lua layer switched on."""
    out = Path(tempfile.gettempdir()) / ("arc3lua_%d.dll" % os.getpid())
    cmd = ["g++", "-O2", "-std=c++17", "-shared", "-pthread", "-w",
           "-DARC3_WITH_LUA",
           "-I", str(LUA_SRC), "-I", str(ROOT / "cpp"),
           "-o", str(out),
           str(ROOT / "cpp" / "arc3_core.cpp"),
           str(ROOT / "cpp" / "lua_one.cpp"),
           str(ROOT / "cpp" / "ag" / "autograd.cpp")]
    if os.name != "nt":
        cmd.insert(1, "-fPIC")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("build failed:\n" + r.stderr[-4000:])
    lib = ctypes.CDLL(str(out))
    lib.arc3_lua_new.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    lib.arc3_lua_new.restype = ctypes.c_int
    lib.arc3_lua_load.argtypes = [ctypes.c_int, ctypes.c_char_p]
    lib.arc3_lua_load.restype = ctypes.c_int
    lib.arc3_lua_observe.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int8), ctypes.c_int]
    lib.arc3_lua_step.argtypes = [ctypes.c_int] + [ctypes.POINTER(ctypes.c_int)] * 3
    lib.arc3_lua_step.restype = ctypes.c_int
    lib.arc3_lua_drain.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    lib.arc3_lua_drain.restype = ctypes.c_int
    return lib


def drain(lib, h) -> str:
    buf = ctypes.create_string_buffer(65536)
    n = lib.arc3_lua_drain(h, buf, len(buf))
    return buf.value.decode("utf-8", "replace") if n else ""


def play(lib, game: str, script: str, max_actions: int, verbose: bool) -> dict:
    quiet()
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    env = arc.make(game)
    obs = env.step(GameAction.RESET)

    avail = [a.value if hasattr(a, "value") else int(a) for a in obs.available_actions]
    arr = (ctypes.c_int * len(avail))(*avail)
    h = lib.arc3_lua_new(arr, len(avail))
    if lib.arc3_lua_load(h, script.encode("utf-8")) != 0:
        return {"game": game, "error": drain(lib, h).strip(), "levels": 0, "actions": 0}

    def observe(o):
        f = frame_of(o)
        if f is None:
            return
        flat = np.ascontiguousarray(np.asarray(f, dtype=np.int8).reshape(-1))
        lib.arc3_lua_observe(h, flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
                             int(o.levels_completed))

    observe(obs)
    a, x, y = (ctypes.c_int(0) for _ in range(3))
    used, err = 0, ""
    while used < max_actions:
        rc = lib.arc3_lua_step(h, ctypes.byref(a), ctypes.byref(x), ctypes.byref(y))
        if rc != 1:
            err = drain(lib, h)
            break
        used += 1
        data = {"x": x.value, "y": y.value} if a.value == 6 else {}
        try:
            obs = env.step(BY_VALUE[a.value], data=data)
        except Exception as e:                    # an in-game fault is the game's, not ours
            err = f"env raised {type(e).__name__}: {e}"
            break
        observe(obs)
        state = _STATE_CODE.get(getattr(obs.state, "value", str(obs.state)), 1)
        if state == 2:
            break
        if verbose and used % 200 == 0:
            print(f"  ...{used} actions, level {obs.levels_completed}", flush=True)

    out = drain(lib, h)
    lib.arc3_lua_free(h)
    return {"game": game, "levels": obs.levels_completed, "actions": used,
            "log": (out + err).strip()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="ka59")
    ap.add_argument("--script", required=True)
    ap.add_argument("--max", type=int, default=2000)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    script = Path(args.script).read_text(encoding="utf-8")
    lib = build()
    r = play(lib, args.game, script, args.max, not args.quiet)
    if r.get("log"):
        print(r["log"])
    print(f"game={r['game']}  levels={r.get('levels', 0)}  actions={r.get('actions', 0)}"
          + (f"  ERROR {r['error']}" if r.get("error") else ""))


if __name__ == "__main__":
    main()
