# xArm7 + Sharpa viewer snapshots

Six `interactive_viewer` snapshots spanning the control run, repaired with
`isaacsimenvs/utils/interactive_viewer/fix_snapshot_urdf.py` — the trainer was
started before the viewer's URDF URL was corrected, so the copies on wandb
still point at the XHand model and fail with
`Joint "left_1_thumb_CMC_FE" not found in URDF`.

| file | roughly |
|---|---|
| `interactive_viewer_0_*` | run start |
| `interactive_viewer_40_*` | learning to lift |
| `interactive_viewer_80_*` | lifting reliably |
| `interactive_viewer_120_*` | goal-reaching |
| `interactive_viewer_160_*` | tolerance curriculum running |
| `interactive_viewer_173_*` | latest |

These pages pull three.js from a CDN and the robot meshes from this repo's raw
URLs, so open them over the network (a `file://` copy works too, given
internet access). GitHub itself serves `.html` as plain text — use a proxy that
sets `text/html`, e.g. replace `raw.githubusercontent.com` with `raw.githack.com`.
