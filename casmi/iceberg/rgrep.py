"""rgrep.py <root> <regex> [glob]: print file:line:text for matches (build-machine helper, read-only)."""
import re, sys, pathlib
root, pat = pathlib.Path(sys.argv[1]), re.compile(sys.argv[2])
glob = sys.argv[3] if len(sys.argv) > 3 else "**/*.py"
for p in sorted(root.glob(glob)):
    try:
        for i, l in enumerate(p.read_text(encoding="utf8", errors="replace").splitlines(), 1):
            if pat.search(l):
                print(f"{p.relative_to(root)}:{i}:{l.strip()[:160]}")
    except OSError:
        pass
