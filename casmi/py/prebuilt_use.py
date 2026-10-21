# PREBUILT-HOOK (inserted by casmi/py/use_prebuilt.py)
# Binaries from the build kernel yomeiotani/casmi26-cpp-build (kernel_sources). Every later
# subprocess.run(["g++", ...]) / (["cmake", ...]) goes through _pb_run: it copies the prebuilt binary
# to the -o path only when
#   - the notebook's own command, normalised, equals the command recorded for that binary,
#   - the main .cpp's sha256 matches,
#   - the tree hash of the mounted rdkit-min/ + autograd/ (fpnet/ minus *.bin for cmake) matches,
#   - the copied binary's sha256 matches and it gives the recorded exit code when run with no args;
# otherwise it prints the reason and runs the original command unchanged. Any other call passes through.
_PB_INPUT = Path(os.environ.get("PB_INPUT", "/kaggle/input"))
_PB_WORK = Path(os.environ.get("PB_WORK", "/kaggle/working"))
if getattr(subprocess.run, "_pb_real", None) is not None:   # cell re-run: do not wrap twice
    subprocess.run = subprocess.run._pb_real
_PB_REAL_RUN = subprocess.run
_PB = {"searched": False, "manifest": None, "dir": None, "tree": {}, "cmake_ok": set(), "log": []}


class _PBMiss(Exception):
    pass


def _pb_manifest():
    if not _PB["searched"]:
        _PB["searched"] = True
        t = time.time()
        for k in range(1, 5):   # at most 4 levels below /kaggle/input, never a full rglob
            for hit in sorted(_PB_INPUT.glob("/".join(["*"] * k + ["bin", "manifest.json"]))):
                try:
                    m = json.loads(hit.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if isinstance(m, dict) and m.get("format") == PB_FORMAT:
                    _PB["manifest"], _PB["dir"] = m, hit.parent
                    break
            if _PB["manifest"] is not None:
                break
        print("PREBUILT manifest:", _PB["dir"], "(%d binaries, %.1fs)"
              % (len(_PB["manifest"]["binaries"]) if _PB["manifest"] else 0, time.time() - t), flush=True)
    if _PB["manifest"] is None:
        raise _PBMiss("no build-kernel manifest under " + str(_PB_INPUT))
    return _PB["manifest"]


def _pb_tree(root, dirs, excl):
    key = (str(root), tuple(dirs), tuple(excl))
    if key not in _PB["tree"]:
        t = time.time()
        sha, n, nb = pb_tree_hash(root, dirs, excl)
        _PB["tree"][key] = sha
        print("PREBUILT tree hash %s %s: %s (%d files, %.1f MB, %.1fs)"
              % (Path(root).name, "+".join(dirs), sha[:16], n, nb / 1e6, time.time() - t), flush=True)
    return _PB["tree"][key]


def _pb_install(e, dest):
    src = _PB["dir"] / e["file"]
    if not src.exists():
        raise _PBMiss("binary file missing: " + str(src))
    if pb_sha_file(src) != e["bin_sha256"]:
        raise _PBMiss("binary sha256 differs from manifest")
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".pbtmp")
    shutil.copyfile(src, tmp)
    os.chmod(tmp, 0o755)
    if e.get("noarg_rc") is not None:
        d = tempfile.mkdtemp(prefix="pb_smoke_")
        try:
            rc = _PB_REAL_RUN([str(tmp)], capture_output=True, cwd=d, timeout=60,
                              stdin=subprocess.DEVNULL).returncode
        except Exception as ex:
            rc = repr(ex)
        shutil.rmtree(d, ignore_errors=True)
        if rc != e["noarg_rc"]:
            os.remove(tmp)
            raise _PBMiss("smoke test: no-arg exit %s, expected %s" % (rc, e["noarg_rc"]))
    os.replace(tmp, dest)


def _pb_gcc(cmd, cwd):
    out = cmd[cmd.index("-o") + 1]
    ds_root = None
    for i, a in enumerate(cmd[:-1]):
        if a == "-I" and Path(cmd[i + 1]).name == "rdkit-min":
            ds_root = Path(cmd[i + 1]).parent
            break
    norm = pb_norm_cmd(cmd, cwd, ds_root, _PB_WORK)
    m = _pb_manifest()
    same_cmd = [e for e in m["binaries"] if e["kind"] == "gcc" and e["cmd"] == norm]
    if not same_cmd:
        keys = {e["key"] for e in m["binaries"]}
        mains = [a for a in norm if a.endswith(".cpp") and a in keys]
        raise _PBMiss("compile command differs from the build kernel's" if mains
                      else "no prebuilt for this source")
    key = same_cmd[0]["key"]
    main_path = next(a for a, n in zip(cmd, norm) if n == key)
    if not Path(main_path).is_absolute() and cwd is not None:
        main_path = str(Path(cwd) / main_path)
    msha = pb_sha_file(main_path)
    cands = [e for e in same_cmd if e["main_sha256"] == msha]
    if not cands:
        raise _PBMiss("main source sha256 differs (%s)" % Path(main_path).name)
    e = cands[0]
    if e["tree_dirs"]:
        if ds_root is None:
            raise _PBMiss("no rdkit-min include dir in the command")
        if _pb_tree(ds_root, e["tree_dirs"], e["tree_exclude"]) != e["tree_sha256"]:
            raise _PBMiss("rdkit-min/autograd tree hash differs (dataset version changed?)")
    _pb_install(e, out)
    return e["name"]


def _pb_cmake_configure(cmd):
    src = Path(cmd[cmd.index("-S") + 1])
    bdir = cmd[cmd.index("-B") + 1]
    norm = pb_norm_cmd(cmd, None, src.parent, _PB_WORK)
    m = _pb_manifest()
    group = [e for e in m["binaries"] if e["kind"] == "cmake" and e["cmd"] == norm]
    if not group:
        raise _PBMiss("no prebuilt for this cmake configure line")
    if _pb_tree(src.parent, group[0]["tree_dirs"], group[0]["tree_exclude"]) != group[0]["tree_sha256"]:
        raise _PBMiss("fpnet tree hash differs")
    for e in group:
        _pb_install(e, Path(bdir) / e["file"])
    _PB["cmake_ok"].add(str(Path(bdir)))
    return "+".join(e["name"] for e in group)


def _pb_done(cmd, kw, msg):
    text = kw.get("text") or kw.get("universal_newlines") or kw.get("encoding")
    empty = "" if text else b""
    return subprocess.CompletedProcess(cmd, 0, empty if kw.get("capture_output") or kw.get("stdout") else None,
                                       empty if kw.get("capture_output") or kw.get("stderr") else None)


def _pb_run(*args, **kw):
    cmd = args[0] if args else kw.get("args")
    if isinstance(cmd, (list, tuple)) and cmd and str(cmd[0]) in ("g++", "cmake"):
        cmd_s = [str(c) for c in cmd]
        label = "?"
        try:
            t = time.time()
            if cmd_s[0] == "g++" and "-o" in cmd_s and "-c" not in cmd_s:
                label = Path(cmd_s[cmd_s.index("-o") + 1]).name
                name = _pb_gcc(cmd_s, kw.get("cwd"))
                msg = "PREBUILT %s: used (%s, %.1fs)" % (label, name, time.time() - t)
                print(msg, flush=True); _PB["log"].append(msg)
                return _pb_done(cmd, kw, msg)
            if cmd_s[0] == "cmake" and "-S" in cmd_s and "-B" in cmd_s:
                label = "cmake " + Path(cmd_s[cmd_s.index("-S") + 1]).name
                name = _pb_cmake_configure(cmd_s)
                msg = "PREBUILT %s: used (%s, %.1fs)" % (label, name, time.time() - t)
                print(msg, flush=True); _PB["log"].append(msg)
                return _pb_done(cmd, kw, msg)
            if cmd_s[0] == "cmake" and "--build" in cmd_s:
                bdir = str(Path(cmd_s[cmd_s.index("--build") + 1]))
                if bdir in _PB["cmake_ok"]:
                    print("PREBUILT cmake --build %s: skipped (prebuilt binaries in place)" % Path(bdir).name, flush=True)
                    return _pb_done(cmd, kw, "")
                return _PB_REAL_RUN(*args, **kw)
        except _PBMiss as ex:
            msg = "PREBUILT %s: miss (%s)" % (label, ex)
            print(msg, flush=True); _PB["log"].append(msg)
        except Exception as ex:
            msg = "PREBUILT %s: miss (error %r)" % (label, ex)
            print(msg, flush=True); _PB["log"].append(msg)
    return _PB_REAL_RUN(*args, **kw)


_pb_run._pb_real = _PB_REAL_RUN
subprocess.run = _pb_run
print("PREBUILT hook installed", flush=True)
