# Outputs

Runtime artifacts from `scripts/train.py`, `scripts/eval.py`, and `scripts/render.py` are written here.

Each run creates a unique timestamped directory under one of:

- `outputs/train/`
- `outputs/eval/`
- `outputs/render/`

Tracked files stay small and human-readable by default:

- run manifests
- resolved config snapshots
- summary JSON files
- rollout traces
- render storyboard previews
