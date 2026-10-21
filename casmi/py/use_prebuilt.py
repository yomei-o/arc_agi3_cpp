"""Prebuilt C++ binaries for the CASMI26 submission notebooks.

  python use_prebuilt.py build-nb [source.ipynb ...]
      writes casmi/notebook/casmi_build.ipynb (+ kernel-metadata-build.json): the build kernel
      yomeiotani/casmi26-cpp-build. The match.cpp sources are taken verbatim from cell 0
      (CPP_SOURCE = r\"\"\"...\"\"\") of each source notebook (default: v33nofp).

  python use_prebuilt.py patch <notebook.ipynb> <kernel-metadata.json> [suffix]
      writes <notebook><suffix>.ipynb and kernel-metadata-<tag><suffix>.json (suffix default _pb):
      one hook cell (prebuilt_common.py + prebuilt_use.py) inserted before the first cell that
      compiles, and yomeiotani/casmi26-cpp-build added to kernel_sources. The original files are
      not touched. Re-patching an already patched notebook replaces the hook cell.

Cell code lives in prebuilt_common.py / prebuilt_build.py / prebuilt_use.py next to this file.
"""
import ast, copy, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NBDIR = HERE.parent / "notebook"
BUILD_KERNEL = "yomeiotani/casmi26-cpp-build"
HOOK_MARK = "# PREBUILT-HOOK"


def cell(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.splitlines(keepends=True)}


def src_of(name):
    return (HERE / name).read_text(encoding="utf-8")


def hook_source():
    common = src_of("prebuilt_common.py")
    use = src_of("prebuilt_use.py")
    assert use.startswith(HOOK_MARK)
    return use.split("\n", 1)[0] + "\n" + common + "\n" + use.split("\n", 1)[1]


def check(nb):
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            ast.parse("".join(c["source"]), filename="cell%d" % i)


def build_nb(sources):
    cells = [cell("# CASMI26 cpp build - see casmi/py/use_prebuilt.py and casmi/prebuilt_STATUS.md\n"
                  "MATCH_SOURCES = []\n")]
    seen = set()
    for s in sources:
        nb = json.loads(Path(s).read_text(encoding="utf-8"))
        c0 = "".join(nb["cells"][0]["source"])
        ns = {}
        exec(compile(c0, "cell0", "exec"), ns)
        text = ns.get("CPP_SOURCE")
        if not isinstance(text, str) or text in seen:
            continue
        seen.add(text)
        cells.append(cell("# match.cpp from %s cell 0, verbatim\n%s\nMATCH_SOURCES.append(CPP_SOURCE)\n"
                          % (Path(s).name, c0.rstrip("\n"))))
    cells.append(cell(src_of("prebuilt_common.py") + "\n" + src_of("prebuilt_build.py")))
    ref = json.loads((NBDIR / "casmi_match_v33nofp.ipynb").read_text(encoding="utf-8"))
    nb = {"cells": cells, "metadata": ref["metadata"], "nbformat": ref["nbformat"],
          "nbformat_minor": ref["nbformat_minor"]}
    check(nb)
    (NBDIR / "casmi_build.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    meta = json.loads((NBDIR / "kernel-metadata-v33.json").read_text(encoding="utf-8"))
    meta.update({"id": BUILD_KERNEL, "title": "CASMI26 cpp build", "code_file": "casmi_build.ipynb",
                 "enable_gpu": False, "enable_internet": False, "kernel_sources": []})
    (NBDIR / "kernel-metadata-build.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print("wrote casmi_build.ipynb (%d match sources) + kernel-metadata-build.json" % len(seen))


def patch(nb_path, meta_path, suffix="_pb"):
    nb_path, meta_path = Path(nb_path), Path(meta_path)
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    cells = [c for c in nb["cells"] if not "".join(c["source"]).startswith(HOOK_MARK)]
    first = next(i for i, c in enumerate(cells) if c["cell_type"] == "code"
                 and ('"g++"' in "".join(c["source"]) or '"cmake"' in "".join(c["source"])))
    cells.insert(first, cell(hook_source()))
    out = copy.deepcopy(nb)
    out["cells"] = cells
    check(out)
    stem = nb_path.stem[:-len(suffix)] if nb_path.stem.endswith(suffix) else nb_path.stem
    out_nb = nb_path.with_name(stem + suffix + ".ipynb")
    out_nb.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    base_id = meta["id"][:-len(suffix.replace("_", "-"))] if meta["id"].endswith(suffix.replace("_", "-")) else meta["id"]
    base_title = meta["title"][:-len(suffix.replace("_", " "))] if meta["title"].endswith(suffix.replace("_", " ")) else meta["title"]
    meta["id"] = base_id + suffix.replace("_", "-")
    meta["title"] = base_title + suffix.replace("_", " ")
    meta["code_file"] = out_nb.name
    ks = list(meta.get("kernel_sources") or [])
    if BUILD_KERNEL not in ks:
        ks.append(BUILD_KERNEL)
    meta["kernel_sources"] = ks
    mstem = meta_path.stem[:-len(suffix)] if meta_path.stem.endswith(suffix) else meta_path.stem
    out_meta = meta_path.with_name(mstem + suffix + ".json")
    out_meta.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print("wrote %s (hook before cell %d) and %s (id %s)" % (out_nb.name, first, out_meta.name, meta["id"]))


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "build-nb":
        build_nb(sys.argv[2:] or [str(NBDIR / "casmi_match_v33nofp.ipynb")])
    elif len(sys.argv) >= 4 and sys.argv[1] == "patch":
        patch(sys.argv[2], sys.argv[3], *(sys.argv[4:5]))
    else:
        print(__doc__)
        sys.exit(1)
