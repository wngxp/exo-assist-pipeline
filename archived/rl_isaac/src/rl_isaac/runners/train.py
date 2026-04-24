from __future__ import annotations

from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any

import torch

from rl_isaac.artifacts import prepare_run_directory, write_json, write_jsonl
from rl_isaac.cli import build_parser
from rl_isaac.config.loader import load_project_config
from rl_isaac.config.schema import ProjectConfig
from rl_isaac.rollout import heuristic_policy, rollout_episode
from rl_isaac.tasks.registry import make_env


def run(config_path: Path) -> Path:
    return run_training(load_project_config(config_path))


def run_training(config: ProjectConfig) -> Path:
    if config.runtime.backend == "scaffold":
        return run_scaffold_training(config)
    if config.runtime.backend == "rsl_rl":
        return run_rsl_rl_training(config)
    raise NotImplementedError(
        f"Unsupported training backend '{config.runtime.backend}'. "
        "Supported backends are 'scaffold' and 'rsl_rl'."
    )


def run_scaffold_training(config: ProjectConfig) -> Path:
    env = make_env(config)
    run_dir = prepare_run_directory(
        config.paths.output_root,
        stage="train",
        experiment_name=config.experiment.name,
    )
    policy = partial(heuristic_policy, action_dim=config.task.action.dimension)
    frames, summary = rollout_episode(env, steps=config.train.rollout_steps, policy=policy)
    write_json(run_dir / "run_manifest.json", _manifest(config, entrypoint="train"))
    write_json(run_dir / "resolved_config.json", config.to_dict())
    write_json(
        run_dir / "policy_stub.json",
        {
            "policy_name": "heuristic_velocity_tracking",
            "notes": "Placeholder policy artifact until RSL-RL integration is added.",
            "action_dimension": config.task.action.dimension,
        },
    )
    write_json(run_dir / "train_summary.json", summary)
    write_jsonl(run_dir / "rollout_trace.jsonl", (frame.to_dict() for frame in frames))
    return run_dir


def run_rsl_rl_training(config: ProjectConfig) -> Path:
    on_policy_runner_cls, tensordict_cls = _import_rsl_rl_dependencies()
    env = make_env(config)
    vec_env = _RslRlVecEnvAdapter(env, config, tensordict_cls=tensordict_cls)
    run_dir = prepare_run_directory(
        config.paths.output_root,
        stage="train",
        experiment_name=config.experiment.name,
    )
    runner_cfg = _build_rsl_rl_runner_config(config)

    write_json(run_dir / "run_manifest.json", _manifest(config, entrypoint="train"))
    write_json(run_dir / "resolved_config.json", config.to_dict())
    write_json(run_dir / "rsl_rl_runner_config.json", runner_cfg)

    runner = on_policy_runner_cls(
        vec_env,
        deepcopy(runner_cfg),
        log_dir=run_dir.as_posix(),
        device=str(vec_env.device),
    )
    runner.learn(num_learning_iterations=config.train.iterations)

    checkpoint_path = run_dir / "policy_final.pt"
    runner.save(checkpoint_path.as_posix())
    write_json(
        run_dir / "train_summary.json",
        {
            "backend": "rsl_rl",
            "iterations_requested": config.train.iterations,
            "iterations_completed": 0
            if config.train.iterations <= 0
            else runner.current_learning_iteration + 1,
            "checkpoint_path": checkpoint_path.as_posix(),
        },
    )
    return run_dir


def main() -> int:
    parser = build_parser("Run the rl_isaac humanoid walking training scaffold.")
    args = parser.parse_args()
    run_dir = run(args.config)
    print(f"Training artifacts written to {run_dir}")
    return 0


def _manifest(config: ProjectConfig, *, entrypoint: str) -> dict[str, Any]:
    return {
        "entrypoint": entrypoint,
        "experiment_name": config.experiment.name,
        "task": config.experiment.task,
        "backend": config.runtime.backend,
        "mode": config.runtime.mode,
    }


def _import_rsl_rl_dependencies() -> tuple[type[Any], type[Any]]:
    try:
        from rsl_rl.runners import OnPolicyRunner
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The 'rsl_rl' backend requires the rsl_rl package in the active Python environment. "
            "Activate the environment that has rsl_rl installed before running training."
        ) from exc

    try:
        from tensordict import TensorDict
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The 'rsl_rl' backend requires tensordict in the active Python environment."
        ) from exc

    return OnPolicyRunner, TensorDict


def _build_rsl_rl_runner_config(config: ProjectConfig) -> dict[str, Any]:
    return {
        "run_name": config.experiment.name,
        "logger": "tensorboard",
        "num_steps_per_env": config.train.rollout_steps,
        "save_interval": config.train.checkpoint_interval,
        "check_for_nan": True,
        "obs_groups": {
            "actor": ["policy"],
            "critic": ["policy"],
        },
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": 4,
            "num_mini_batches": 4,
            "clip_param": 0.2,
            "gamma": 0.99,
            "lam": 0.95,
            "value_loss_coef": 1.0,
            "entropy_coef": 0.01,
            "learning_rate": 3e-4,
            "max_grad_norm": 1.0,
            "use_clipped_value_loss": True,
            "schedule": "adaptive",
            "desired_kl": 0.01,
            "normalize_advantage_per_mini_batch": False,
            "rnd_cfg": None,
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [256, 256, 256],
            "activation": "elu",
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [256, 256, 256],
            "activation": "elu",
        },
    }


class _RslRlVecEnvAdapter:
    """Minimal single-environment adapter for rsl_rl's OnPolicyRunner."""

    def __init__(self, env: Any, config: ProjectConfig, *, tensordict_cls: type[Any]) -> None:
        self._env = env
        self._tensordict_cls = tensordict_cls
        self.device = torch.device(config.runtime.device)
        self.cfg = config.to_dict()
        self.num_envs = 1
        self.num_actions = config.task.action.dimension
        self.max_episode_length = config.task.termination.max_steps
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_reward = 0.0
        self._latest_observation = env.reset(seed=config.experiment.seed)
        self._observation_keys = tuple(self._latest_observation.keys())

    def get_observations(self) -> Any:
        return self._tensordict_cls(
            {"policy": self._observation_tensor(self._latest_observation)},
            batch_size=[self.num_envs],
            device=self.device,
        )

    def step(self, actions: torch.Tensor) -> tuple[Any, torch.Tensor, torch.Tensor, dict[str, Any]]:
        action_tensor = actions[0] if actions.ndim == 2 else actions
        result = self._env.step(action_tensor.detach().to("cpu").tolist())

        reward_value = float(result.reward)
        done_value = float(result.terminated)
        self._episode_reward += reward_value
        self.episode_length_buf += 1

        rewards = torch.tensor([reward_value], dtype=torch.float32, device=self.device)
        dones = torch.tensor([done_value], dtype=torch.float32, device=self.device)
        extras: dict[str, Any] = {
            "time_outs": torch.tensor(
                [1.0 if result.termination_reason == "max_steps" else 0.0],
                dtype=torch.float32,
                device=self.device,
            )
        }

        if result.terminated:
            extras["episode"] = {
                "reward": torch.tensor([self._episode_reward], dtype=torch.float32, device=self.device),
                "length": self.episode_length_buf.to(dtype=torch.float32).clone(),
            }
            self._latest_observation = self._env.reset()
            self._episode_reward = 0.0
            self.episode_length_buf.zero_()
        else:
            self._latest_observation = result.observation

        return self.get_observations(), rewards, dones, extras

    def _observation_tensor(self, observation: dict[str, float]) -> torch.Tensor:
        values = [float(observation[key]) for key in self._observation_keys]
        return torch.tensor(values, dtype=torch.float32, device=self.device).unsqueeze(0)
