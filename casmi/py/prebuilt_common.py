# Shared by the build kernel (prebuilt_build.py) and the submission hook (prebuilt_use.py).
# use_prebuilt.py pastes this file in front of each of them, so a notebook cell is self-contained.
#
# TREE HASH: sha256 over every file under <root>/<d> for d in dirs, sorted by its relative path
# written with '/', each file contributing (path, NUL, byte length, NUL, raw bytes). Binary reads
# and '/'-joined paths make it the same number on Windows and Linux.
import hashlib, json, os, shutil, subprocess, sys, tempfile, time
from pathlib import Path

PB_FORMAT = "casmi26-prebuilt-v1"


def pb_tree_hash(root, dirs, exclude_suffixes=()):
    root = Path(root)
    excl = tuple(exclude_suffixes)
    items = []
    for d in dirs:
        base = root / d
        if not base.is_dir():
            items.append((d + "/<missing>", None))
            continue
        for dp, dns, fns in os.walk(base):
            rel_dp = Path(dp).relative_to(root).as_posix()
            for fn in fns:
                if excl and fn.endswith(excl):
                    continue
                items.append((rel_dp + "/" + fn, os.path.join(dp, fn)))
    items.sort(key=lambda x: x[0])
    h = hashlib.sha256()
    nbytes = 0
    for rel, full in items:
        h.update(rel.encode("utf-8") + b"\0")
        if full is None:
            h.update(b"<missing>\0")
            continue
        with open(full, "rb") as f:
            data = f.read()
        h.update(str(len(data)).encode() + b"\0")
        h.update(data)
        nbytes += len(data)
    return h.hexdigest(), len(items), nbytes


def pb_sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def pb_norm_cmd(cmd, cwd, ds_root, work):
    """A compile command with the machine-specific parts replaced: the dataset root by <DS>, the
    working dir by <WORK>, the output (-o / -B) by <OUT>; relative .cpp args resolved against cwd."""
    pre = []
    if ds_root is not None:
        pre.append((Path(ds_root).as_posix().rstrip("/"), "<DS>"))
    if work is not None:
        pre.append((Path(work).as_posix().rstrip("/"), "<WORK>"))
    out, prev = [], None
    for a in cmd:
        a = str(a)
        if prev in ("-o", "-B"):
            out.append("<OUT>")
        elif a.startswith("-") or ("/" not in a and "\\" not in a and not a.endswith(".cpp")):
            out.append(a)
        else:
            p = Path(a)
            if not p.is_absolute() and cwd is not None and a.endswith(".cpp"):
                p = Path(cwd) / p
            s = p.as_posix()
            for k, v in pre:
                if s == k or s.startswith(k + "/"):
                    s = v + s[len(k):]
                    break
            out.append(s)
        prev = a
    return out
