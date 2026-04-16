from __future__ import annotations

import sys
from pathlib import Path

import imageio.v2 as imageio
import myosuite  # noqa: F401
import numpy as np
from myosuite.utils import gym as myogym

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rl.baselines.load_deprl_reference import load_deprl_reference, wrap_deprl_env

ENV_ID = "myoLegWalk-v0"
FPS = 30
MAX_EPISODE_LENGTH = 1000
DEFAULT_WIDTH = 960
DEFAULT_HEIGHT = 720
OUTPUT_PATH = Path("/Users/wxp/dev/exo-assist-pipeline/rl/rl_output/stage1_reference.mp4")


def _normalize_step(result):
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return obs, reward, bool(terminated or truncated), info

    if len(result) == 4:
        obs, reward, done, info = result
        return obs, reward, bool(done), info

    raise ValueError(f"Unexpected step result length: {len(result)}")


def _frame_to_uint8(frame):
    frame = np.asarray(frame)
    if frame.dtype == np.uint8:
        return frame
    if np.issubdtype(frame.dtype, np.floating) and frame.size > 0 and np.nanmax(frame) <= 1.0:
        frame = frame * 255.0
    return np.clip(frame, 0, 255).astype(np.uint8)


def find_simulator(root_env):
    seen = set()
    stack = [root_env]

    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))

        sim = getattr(current, "sim", None)
        if sim is not None:
            model = getattr(sim, "model", None)
            data = getattr(sim, "data", None)
            if model is not None and data is not None:
                return sim, model, data

        for attr in ("env", "wrapped_env", "base_env", "unwrapped"):
            child = getattr(current, attr, None)
            if child is not None and child is not current:
                stack.append(child)

    raise RuntimeError(
        "Could not access MuJoCo simulator/model/data from the environment wrappers."
    )


def resolve_render_size(model):
    width = DEFAULT_WIDTH
    height = DEFAULT_HEIGHT

    vis = getattr(model, "vis", None)
    if vis is not None:
        global_vis = getattr(vis, "global_", None)
        if global_vis is not None:
            offwidth = int(getattr(global_vis, "offwidth", 0) or 0)
            offheight = int(getattr(global_vis, "offheight", 0) or 0)
            if offwidth > 0:
                width = offwidth
            if offheight > 0:
                height = offheight

    return width, height


def build_offscreen_renderer(sim, model, data, width, height):
    if hasattr(sim, "render"):
        def render_frame():
            try:
                frame = sim.render(width=width, height=height, mode="offscreen")
            except TypeError:
                frame = sim.render(width=width, height=height)
            return _frame_to_uint8(frame)

        return render_frame

    try:
        import mujoco
    except ModuleNotFoundError:
        mujoco = None

    if mujoco is not None:
        renderer = mujoco.Renderer(model, height=height, width=width)

        def render_frame():
            renderer.update_scene(data)
            frame = renderer.render()
            return _frame_to_uint8(frame)

        return render_frame

    try:
        import mujoco_py
    except ModuleNotFoundError:
        mujoco_py = None

    if mujoco_py is not None:
        context = mujoco_py.MjRenderContextOffscreen(sim, device_id=-1)
        if hasattr(sim, "add_render_context"):
            sim.add_render_context(context)

        def render_frame():
            context.render(width, height)
            frame = context.read_pixels(width, height, depth=False)
            return _frame_to_uint8(np.flipud(frame))

        return render_frame

    raise RuntimeError(
        "Could not create a MuJoCo offscreen renderer. "
        "Tried simulator.render(), mujoco.Renderer, and mujoco_py offscreen context."
    )


def run_episode(model, env, render_frame):
    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    frames = [render_frame()]
    done = False
    steps = 0

    while not done and steps < MAX_EPISODE_LENGTH:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = _normalize_step(env.step(action))
        del reward, info
        steps += 1
        frames.append(render_frame())

    terminated_early = done
    print(f"Episode length: {steps}")
    print(f"Terminated early: {terminated_early}")
    return frames, steps, terminated_early


def save_video(frames):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(OUTPUT_PATH, fps=FPS) as writer:
        for frame in frames:
            writer.append_data(frame)
    print(f"Saved video to: {OUTPUT_PATH}")


def main():
    base_env = myogym.make(ENV_ID, reset_type="random")
    env = wrap_deprl_env(base_env)

    try:
        model = load_deprl_reference(env)
        sim, model_mj, data_mj = find_simulator(env)
        width, height = resolve_render_size(model_mj)
        render_frame = build_offscreen_renderer(sim, model_mj, data_mj, width, height)

        frames, steps, terminated_early = run_episode(model, env, render_frame)
        del steps, terminated_early

        if not frames:
            raise RuntimeError("No RGB frames were captured from the MuJoCo renderer.")

        save_video(frames)
    finally:
        env.close()


if __name__ == "__main__":
    main()
