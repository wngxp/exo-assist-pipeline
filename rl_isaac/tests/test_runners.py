from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
import sys
from tempfile import TemporaryDirectory


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rl_isaac.config.loader import load_project_config
from rl_isaac.runners.eval import run_evaluation
from rl_isaac.runners.render import run_render
from rl_isaac.runners.train import run_training


class RunnerSmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "experiments" / "humanoid_walk.toml"
        self.base_config = load_project_config(config_path)

    def test_train_eval_and_render_write_artifacts(self) -> None:
        with TemporaryDirectory() as tmpdir:
            temp_output_root = Path(tmpdir)
            config = replace(
                self.base_config,
                paths=replace(self.base_config.paths, output_root=temp_output_root),
            )

            train_dir = run_training(config)
            eval_dir = run_evaluation(config)
            render_dir = run_render(config)

            self.assertTrue((train_dir / "train_summary.json").exists())
            self.assertTrue((train_dir / "policy_stub.json").exists())
            self.assertTrue((eval_dir / "eval_summary.json").exists())
            self.assertTrue((render_dir / "storyboard.txt").exists())


if __name__ == "__main__":
    unittest.main()
