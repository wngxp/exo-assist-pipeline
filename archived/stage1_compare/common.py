from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RL_DIR = REPO_ROOT / "rl"
RL_OUTPUT_DIR = RL_DIR / "rl_output"
STAGE1_COMPARE_DIR = RL_DIR / "stage1_compare"
RESULTS_DIR = RL_OUTPUT_DIR / "stage1" / "compare"


def ensure_results_dir() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def existing_checkpoint_path(base_path: Path) -> Path | None:
    candidates = [base_path]
    if base_path.suffix != ".zip":
        candidates.append(base_path.with_suffix(".zip"))
        candidates.append(Path(f"{base_path}.zip"))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def first_existing_path(candidates: list[Path], *, label: str) -> Path:
    for candidate in candidates:
        resolved = existing_checkpoint_path(candidate)
        if resolved is not None:
            return resolved
    candidate_text = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"Could not find {label}. Checked:\n{candidate_text}")


def unique_output_path(filename: str) -> Path:
    ensure_results_dir()
    target = RESULTS_DIR / filename
    if not target.exists():
        return target

    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    return target.with_name(f"{target.stem}__{timestamp}{target.suffix}")


def latest_result_path(filename: str) -> Path | None:
    ensure_results_dir()
    target = RESULTS_DIR / filename
    matches = sorted(
        RESULTS_DIR.glob(f"{target.stem}*{target.suffix}"),
        key=lambda path: path.stat().st_mtime,
    )
    if not matches:
        return None
    return matches[-1]


def write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def write_episode_lengths_csv(path: Path, episode_lengths: list[int]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["episode", "length"])
        for idx, length in enumerate(episode_lengths, start=1):
            writer.writerow([idx, int(length)])
    return path


def build_length_summary(
    *,
    evaluation_name: str,
    model_path: Path,
    episode_lengths: list[int],
    num_episodes: int,
    deterministic: bool = True,
    extra: dict | None = None,
) -> dict:
    if not episode_lengths:
        raise ValueError("episode_lengths must not be empty")

    payload = {
        "evaluation_name": evaluation_name,
        "model_path": str(model_path),
        "num_episodes": int(num_episodes),
        "deterministic": bool(deterministic),
        "episode_lengths": [int(length) for length in episode_lengths],
        "mean_episode_length": float(sum(episode_lengths) / len(episode_lengths)),
        "min_episode_length": int(min(episode_lengths)),
        "max_episode_length": int(max(episode_lengths)),
        "all_episode_lengths_identical": len(set(episode_lengths)) == 1,
        "saved_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    if extra:
        payload.update(extra)
    return payload


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_text(value) -> str:
    if value is None:
        return "MISSING"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)
