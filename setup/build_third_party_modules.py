"""Build src/modules/{rots,owmd,wmdo}.py from upstream code.

Takes the line ranges listed in patches/third_party_modules.json from the
upstream repositories cloned under third_party/. Our own changes are added
afterwards by patches/third_party_modules.patch.
"""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
spec = json.loads((root / "patches/third_party_modules.json").read_text())
for output, ranges in spec.items():
    lines = []
    for src, start, end in ranges:
        src_lines = (root / "third_party" / src).read_text().splitlines()
        lines.extend(src_lines[start - 1:end])
    (root / output).write_text("".join(line + "\n" for line in lines))
