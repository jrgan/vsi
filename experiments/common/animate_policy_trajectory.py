"""Render saved trajectory values and onward decompositions as a GIF."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from experiments.common.io import load_final_value_sets, load_trajectory
from experiments.common.animate_value_sets import _outline


def _decompose(run_dir: Path, states: np.ndarray, achieved: np.ndarray) -> dict[str, np.ndarray]:
    """Find one feasible onward decomposition for each saved achieved value."""
    from experiments.scmdp.config import ExperimentConfig as SCMDPConfig
    from experiments.scmdp.instance import load_instance as load_scmdp
    from experiments.scmdp.runner import SCMDPExperimentRunner
    from experiments.markovgame.config import ExperimentConfig as MGConfig
    from experiments.markovgame.instance import load_instance as load_mg
    from experiments.markovgame.runner import MGExperimentRunner
    from vsi.vertex_rep import VertexSets

    raw_config = json.loads((run_dir / "config.json").read_text())
    if "num_agents" in raw_config["instance"]:
        config = MGConfig.from_dict(raw_config)
        instance = load_mg(run_dir)
        experiment = MGExperimentRunner(config)
    else:
        config = SCMDPConfig.from_dict(raw_config)
        instance = load_scmdp(run_dir)
        experiment = SCMDPExperimentRunner(config)

    sets = VertexSets(load_final_value_sets(run_dir))
    grid = experiment.make_grid(instance)
    engine = experiment.make_engine(instance, grid, config.algorithm.mode)
    onward_steps: list[int] = []
    onward_states: list[int] = []
    onward_actions: list[int] = []
    onward_values: list[np.ndarray] = []

    for step, (state, value) in enumerate(zip(states, achieved)):
        expansion = engine.expand_value(value, int(state), sets)
        if expansion is None:
            raise ValueError(f"No feasible decomposition for saved trajectory step {step}")
        for outcome, onward_policy in expansion.onward_policies.items():
            action, state = int(outcome.action), int(outcome.state)
            # Markov-game expansions also include off-path values for player
            # deviations. Only the recommended action actually played belongs
            # to the policy rollout shown here.
            if (getattr(outcome, "action_played", action) != action or
                    expansion.action_probs[action] <= 1e-8 or
                    instance.transitions[states[step], action, state] <= 1e-12):
                continue
            onward_steps.append(step)
            onward_states.append(state)
            onward_actions.append(action)
            onward_values.append(np.asarray(onward_policy.value, dtype=float).copy())

    return {
        "cache_version": np.array(2),
        "states": states,
        "achieved": achieved,
        "onward_steps": np.asarray(onward_steps, dtype=int),
        "onward_states": np.asarray(onward_states, dtype=int),
        "onward_actions": np.asarray(onward_actions, dtype=int),
        "onward_values": np.stack(onward_values) if onward_values else np.empty((0, achieved.shape[1])),
    }


def animate(run_dir: Path, output: Path, duration_ms: int = 1800) -> None:
    trajectory = load_trajectory(run_dir)
    if not trajectory:
        raise ValueError(f"No saved trajectory in {run_dir}")
    states = np.asarray([state for state, _ in trajectory], dtype=int)
    achieved = np.stack([value for _, value in trajectory])
    sets = load_final_value_sets(run_dir)
    cache = run_dir / "policy_expansions.npz"
    if cache.exists():
        with np.load(cache) as data:
            replay = {key: data[key].copy() for key in data.files}
        if (not np.array_equal(replay["states"], states) or
                not np.allclose(replay["achieved"], achieved, atol=1e-5, rtol=1e-5)):
            raise ValueError(f"Cached expansions do not match {run_dir / 'trajectory.npz'}")
    if not cache.exists() or int(replay.get("cache_version", 0)) < 2:
        replay = _decompose(run_dir, states, achieved)
        np.savez(cache, **replay)
        print(f"Saved reconstructed policy expansions to {cache}")

    num_states = len(sets)
    outlines = [_outline(points) for points in sets]
    all_points = [points for points in outlines if len(points)]
    all_points += [achieved[:, :2], replay.get("targets", achieved)[:, :2],
                   replay["onward_values"][:, :2]]
    bounds = np.concatenate([points for points in all_points if len(points)])
    low, high = bounds.min(axis=0), bounds.max(axis=0)
    padding = np.maximum((high - low) * 0.055, 0.05)
    xlim = (low[0] - padding[0], high[0] + padding[0])
    ylim = (low[1] - padding[1], high[1] + padding[1])

    # Match the value-set GIF's fixed, four-panel layout and text sizes.
    fig, axes = plt.subplots(1, num_states, figsize=(4 * num_states, 4.6), dpi=150,
                             squeeze=False, sharex=True, sharey=True)
    fig.subplots_adjust(left=0.045, right=0.995, bottom=0.19, top=0.81, wspace=0.30)
    title = fig.suptitle("", fontsize=18)
    frames: list[Image.Image] = []
    try:
        for step in range(len(states)):
            for phase in ("target", "onward"):
                for state, ax in enumerate(axes.flat[:num_states]):
                    ax.clear()
                    ax.set_xlim(*xlim)
                    ax.set_ylim(*ylim)
                    ax.set_aspect("equal", adjustable="box")
                    ax.set_anchor("C")
                    ax.set_title(f"State {state}", fontsize=14.4)
                    ax.set_xlabel("Dimension 1", fontsize=12)
                    ax.set_ylabel("Dimension 2", fontsize=12)
                    ax.grid(alpha=0.2)
                    ax.tick_params(labelbottom=True, labelleft=True, labelsize=12)

                    # Use State 0's policy_trajectory.pdf colour for every value set.
                    points = outlines[state]
                    color = plt.cm.tab10.colors[0]
                    if len(points) >= 3:
                        ax.fill(points[:, 0], points[:, 1], color=color, alpha=0.15)
                        closed = np.vstack((points, points[0]))
                        ax.plot(closed[:, 0], closed[:, 1], color=color, lw=1.0)
                    elif len(points) == 2:
                        ax.plot(points[:, 0], points[:, 1], color=color, lw=1.0)
                    if state == states[step]:
                        target = replay.get("targets", achieved)[step]
                        target_right = target[0] > (xlim[0] + xlim[1]) / 2
                        ax.scatter(target[0], target[1], c="red", s=80,
                                   edgecolors="darkred", linewidths=1.44, zorder=7)
                        ax.annotate("target", target[:2], fontsize=11,
                                    ha="right" if target_right else "left", va="bottom",
                                    xytext=(-7 if target_right else 7, 8),
                                    textcoords="offset points", color="red", zorder=8)

                    if phase == "onward":
                        mask = ((replay["onward_steps"] == step) &
                                (replay["onward_states"] == state))
                        for value in replay["onward_values"][mask]:
                            ax.scatter(value[0], value[1], c="blue", s=55,
                                       marker="o", edgecolors="white",
                                       linewidths=0.96, zorder=9)
                            ax.annotate("onward", value[:2], fontsize=9,
                                        ha="right", va="center", xytext=(-8, 0),
                                        textcoords="offset points", color="blue", zorder=10)

                title.set_text(f"Policy expansion {step + 1:0{len(str(len(states)))}d}"
                               f"/{len(states)}")
                fig.canvas.draw()
                frame = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB")
                frames.append(frame.quantize(colors=128))
    finally:
        plt.close(fig)

    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output, save_all=True, append_images=frames[1:],
                   duration=[duration_ms] * (len(frames) - 1) + [5400],
                   loop=0, optimize=True, disposal=2)
    print(f"Saved {len(frames)} frames to {output}")
