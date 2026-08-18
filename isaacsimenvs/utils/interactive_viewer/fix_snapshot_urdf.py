"""Repoint an already-written interactive_viewer snapshot at a different URDF.

The viewer's URDF URL is baked in when the snapshot is written, so a run that
started before the URL was corrected keeps emitting snapshots that fail with
`Joint "<name>" not found in URDF`. Rather than restart training, rewrite the
URL in the saved HTML.

    python fix_snapshot_urdf.py OUT_DIR SNAPSHOT.html [SNAPSHOT.html ...]
"""
import sys
from pathlib import Path

OLD = "https://raw.githubusercontent.com/Yuan-Xinyi/simtoolreal/xarm7-xhand/assets/urdf/xarm7_xhand/xarm7_xhand.urdf"
NEW = "https://raw.githubusercontent.com/Yuan-Xinyi/simtoolreal/xarm7-sharpa/assets/urdf/xarm7_sharpa/xarm7_sharpa.urdf"

out_dir = Path(sys.argv[1])
out_dir.mkdir(parents=True, exist_ok=True)
for src in sys.argv[2:]:
    p = Path(src)
    text = p.read_text()
    if OLD not in text:
        print(f"skip (no stale URL): {p.name}")
        continue
    dst = out_dir / p.name
    dst.write_text(text.replace(OLD, NEW))
    print(f"fixed -> {dst}")
