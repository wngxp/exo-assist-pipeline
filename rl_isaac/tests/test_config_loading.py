from __future__ import annotations

import unittest
from pathlib import Path
import sys


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rl_isaac.config.loader import load_project_config


class ConfigLoadingTest(unittest.TestCase):
    def test_loads_default_experiment_config(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "experiments" / "humanoid_walk.toml"
        config = load_project_config(config_path)

        self.assertEqual(config.experiment.task, "humanoid_walk")
        self.assertEqual(config.task.name, "humanoid_walk")
        self.assertEqual(config.task.action.dimension, len(config.task.action.joint_names))
        self.assertEqual(config.paths.output_root.name, "outputs")


if __name__ == "__main__":
    unittest.main()
