from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rl.stage1_compare.common import latest_result_path, load_json, metric_text


def load_result(filename: str) -> dict | None:
    path = latest_result_path(filename)
    if path is None:
        return None

    payload = load_json(path)
    payload["_result_path"] = str(path)
    return payload


def print_comparison(results: dict[str, dict | None]) -> None:
    headers = ["metric", "simple", "v2_final", "v2_best"]
    rows = [
        (
            "result file",
            results["simple"].get("_result_path") if results["simple"] else None,
            results["v2_final"].get("_result_path") if results["v2_final"] else None,
            results["v2_best"].get("_result_path") if results["v2_best"] else None,
        ),
        (
            "mean episode length",
            results["simple"].get("mean_episode_length") if results["simple"] else None,
            results["v2_final"].get("mean_episode_length") if results["v2_final"] else None,
            results["v2_best"].get("mean_episode_length") if results["v2_best"] else None,
        ),
        (
            "min/max",
            (
                f"{results['simple']['min_episode_length']}/{results['simple']['max_episode_length']}"
                if results["simple"]
                else None
            ),
            (
                f"{results['v2_final']['min_episode_length']}/{results['v2_final']['max_episode_length']}"
                if results["v2_final"]
                else None
            ),
            (
                f"{results['v2_best']['min_episode_length']}/{results['v2_best']['max_episode_length']}"
                if results["v2_best"]
                else None
            ),
        ),
        (
            "all lengths identical",
            results["simple"].get("all_episode_lengths_identical") if results["simple"] else None,
            results["v2_final"].get("all_episode_lengths_identical") if results["v2_final"] else None,
            results["v2_best"].get("all_episode_lengths_identical") if results["v2_best"] else None,
        ),
    ]

    widths = []
    for column_idx, header in enumerate(headers):
        column_values = [header]
        for row in rows:
            column_values.append(metric_text(row[column_idx]))
        widths.append(max(len(value) for value in column_values))

    header_line = " | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers))
    separator = "-+-".join("-" * width for width in widths)
    print(header_line)
    print(separator)

    for row in rows:
        rendered = [metric_text(value).ljust(widths[idx]) for idx, value in enumerate(row)]
        print(" | ".join(rendered))


if __name__ == "__main__":
    results = {
        "simple": load_result("simple_1p5M_eval.json"),
        "v2_final": load_result("v2_final_eval.json"),
        "v2_best": load_result("v2_best_eval.json"),
    }
    print_comparison(results)
