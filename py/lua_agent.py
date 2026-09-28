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
    """Compile the core with the Lua layer switched on.

    Keyed by the content of the sources, so an unchanged tree is compiled once
    and every later run loads it. Ninety seconds a run is not much until it is
    every run of a day spent changing one prompt at a time.
    """
    import hashlib
    src = [ROOT / "cpp" / "arc3_core.cpp", ROOT / "cpp" / "arc3_lua.h",
           ROOT / "cpp" / "arc3_net.h", ROOT / "cpp" / "lua_one.cpp",
           ROOT / "cpp" / "ag" / "autograd.cpp", ROOT / "cpp" / "ag" / "autograd.h"]
    h = hashlib.sha1()
    for f in src:
        if f.exists():
            h.update(f.read_bytes())
    tag = h.hexdigest()[:12]
    cached = Path(tempfile.gettempdir()) / ("arc3lua_%s.dll" % tag)
    if cached.exists():
        return _bind(ctypes.CDLL(str(cached)))

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
    try:
        out.replace(cached)          # publish atomically for the next run
        out = cached
    except OSError:
        pass                         # another process got there first
    return _bind(ctypes.CDLL(str(out)))


def _bind(lib: ctypes.CDLL) -> ctypes.CDLL:
    lib.arc3_lua_new.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    lib.arc3_lua_new.restype = ctypes.c_int
    lib.arc3_lua_load.argtypes = [ctypes.c_int, ctypes.c_char_p]
    lib.arc3_lua_load.restype = ctypes.c_int
    lib.arc3_lua_observe.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int8),
                                     ctypes.c_int, ctypes.c_int]
    lib.arc3_lua_step.argtypes = [ctypes.c_int] + [ctypes.POINTER(ctypes.c_int)] * 3
    lib.arc3_lua_step.restype = ctypes.c_int
    lib.arc3_lua_drain.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    lib.arc3_lua_drain.restype = ctypes.c_int
    return lib


def drain(lib, h) -> str:
    buf = ctypes.create_string_buffer(65536)
    n = lib.arc3_lua_drain(h, buf, len(buf))
    return buf.value.decode("utf-8", "replace") if n else ""


class Session:
    """One game, one host, many scripts.

    The model cannot write a policy for a game it has not seen played. So the
    game and everything the host has learned about it - the avatar, the buttons,
    the walls, the tabular policy's counters - outlive any one script, and the
    model writes a few moves at a time with the board in front of it.
    """

    def __init__(self, lib, game: str):
        quiet()
        self.lib = lib
        self.game = game
        self.arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
        self.env = self.arc.make(game)
        self.obs = self.env.step(GameAction.RESET)
        self.avail = [a.value if hasattr(a, "value") else int(a)
                      for a in self.obs.available_actions]
        arr = (ctypes.c_int * len(self.avail))(*self.avail)
        self.h = lib.arc3_lua_new(arr, len(self.avail))
        self.used = 0
        self.done = False
        self._observe()

    def _observe(self):
        f = frame_of(self.obs)
        if f is None:
            return
        flat = np.ascontiguousarray(np.asarray(f, dtype=np.int8).reshape(-1))
        st = _STATE_CODE.get(getattr(self.obs.state, "value", str(self.obs.state)), 1)
        self.lib.arc3_lua_observe(
            self.h, flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            int(self.obs.levels_completed), st)

    @property
    def levels(self) -> int:
        return int(self.obs.levels_completed)

    def run(self, script: str, budget: int) -> dict:
        """Load a script and let it spend at most `budget` actions."""
        if self.lib.arc3_lua_load(self.h, script.encode("utf-8")) != 0:
            return {"spent": 0, "log": drain(self.lib, self.h).strip(), "loaded": False}
        a, x, y = (ctypes.c_int(0) for _ in range(3))
        spent = 0
        while spent < budget:
            rc = self.lib.arc3_lua_step(self.h, ctypes.byref(a), ctypes.byref(x),
                                        ctypes.byref(y))
            if rc != 1:
                break
            spent += 1
            self.used += 1
            data = {"x": x.value, "y": y.value} if a.value == 6 else {}
            try:
                self.obs = self.env.step(BY_VALUE[a.value], data=data)
            except Exception as e:
                return {"spent": spent, "log": f"the game raised {type(e).__name__}: {e}",
                        "loaded": True}
            self._observe()
            st = _STATE_CODE.get(getattr(self.obs.state, "value", str(self.obs.state)), 1)
            if st == 2:
                self.done = True
                break
        return {"spent": spent, "log": drain(self.lib, self.h).strip(), "loaded": True}

    def close(self):
        self.lib.arc3_lua_free(self.h)


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
        st = _STATE_CODE.get(getattr(o.state, "value", str(o.state)), 1)
        lib.arc3_lua_observe(h, flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
                             int(o.levels_completed), st)

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
    ap.add_argument("--game", default="ka59",
                    help="one id, a comma list, or 'all' for every public game")
    ap.add_argument("--script", required=True)
    ap.add_argument("--max", type=int, default=2000)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    script = Path(args.script).read_text(encoding="utf-8")
    lib = build()

    if args.game == "all":
        quiet()
        arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
        games = sorted({e.game_id.split("-")[0] for e in arc.get_environments()})
    else:
        games = [g.strip() for g in args.game.split(",")]

    # One game tells you nothing. Every remedy that was wrong this month looked
    # right on the game it was found on.
    rows = []
    for g in games:
        try:
            r = play(lib, g, script, args.max, False)
        except Exception as e:
            r = {"game": g, "levels": 0, "actions": 0, "error": f"{type(e).__name__}: {e}"}
        rows.append(r)
        if len(games) == 1 and r.get("log"):
            print(r["log"])
        print(f"  {g:6} levels={r.get('levels', 0)} actions={r.get('actions', 0)}"
              + (f"  ERROR {str(r.get('error'))[:70]}" if r.get("error") else ""), flush=True)

    if len(rows) > 1:
        won = [r for r in rows if r.get("levels", 0) > 0]
        print(f"\n{len(won)}/{len(rows)} ゲームで1レベル以上: "
              + ", ".join(f"{r['game']}({r['levels']}lv/{r['actions']}手)" for r in won))


if __name__ == "__main__":
    main()
