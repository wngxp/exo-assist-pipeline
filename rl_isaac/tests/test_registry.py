from __future__ import annotations

import unittest
from pathlib import Path
import sys


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rl_isaac.config.loader import load_project_config
from rl_isaac.tasks.registry import TASK_REGISTRY, make_env


class TaskRegistryTest(unittest.TestCase):
    def test_humanoid_walk_is_registered(self) -> None:
        self.assertIn("humanoid_walk", TASK_REGISTRY)

    def test_registry_builds_env(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "experiments" / "humanoid_walk.toml"
        config = load_project_config(config_path)
        env = make_env(config)
        self.assertEqual(env.config.name, "humanoid_walk")


if __name__ == "__main__":
    unittest.main()
