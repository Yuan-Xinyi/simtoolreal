"""Watch reset-time object behavior in an Isaac Sim window (issue #17).

Runs zero actions and forces a full reset every ``--reset_every`` steps so you
can visually check whether objects still launch/bounce off the table at reset.
After each reset the object speed at the first physics step is printed per env
(free fall is ~0.16 m/s; a depenetration launch shows up as >~2 m/s and is
flagged). Launch-prone envs are also listed so you can watch them in the grid.

Current defaults use all three mitigations (reset_object_clearance,
reset_avoid_hand, object_max_depenetration_velocity=1.0). Pass ``--legacy`` to
restore the pre-fix behavior (no clearance, no hand rejection, depenetration
cap 1000) for a side-by-side comparison — expect visible launches there.

Example:
    python isaacsimenvs/tests/visualize_reset_launch.py --num_envs 16
    python isaacsimenvs/tests/visualize_reset_launch.py --num_envs 16 --legacy
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--num_assets_per_type", type=int, default=4)
    parser.add_argument("--reset_every", type=int, default=60, help="policy steps between forced resets")
    parser.add_argument("--rounds", type=int, default=0, help="reset windows to run; 0 = until the window closes")
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="pre-fix behavior: no clearance, no hand rejection, depenetration cap 1000",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app_launcher = AppLauncher(args)
    app = app_launcher.app

    import gymnasium as gym
    import torch

    import isaacsimenvs  # noqa: F401  (registers gym envs)
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg import SimToolRealEnvCfg

    cfg = SimToolRealEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.assets.num_assets_per_type = args.num_assets_per_type
    if args.legacy:
        cfg.reset.reset_object_clearance = None
        cfg.reset.reset_avoid_hand = False
        cfg.assets.object_max_depenetration_velocity = 1000.0

    env = gym.make("Isaacsimenvs-SimToolReal-Direct-v0", cfg=cfg)
    inner = env.unwrapped
    zero_action = torch.zeros(
        (inner.num_envs, inner.cfg.action_space), device=inner.device, dtype=torch.float32
    )

    mode = "LEGACY (pre-fix)" if args.legacy else "current defaults"
    print(f"\n[reset-viz] mode: {mode}")
    print(f"[reset-viz] clearance={inner.cfg.reset.reset_object_clearance}  "
          f"avoid_hand={inner.cfg.reset.reset_avoid_hand}  "
          f"depen_cap={inner.cfg.assets.object_max_depenetration_velocity}")
    print(f"[reset-viz] forced reset every {args.reset_every} steps; "
          f"free-fall speed after 1 step ~ {9.81 / 60:.2f} m/s")
    print("[reset-viz] Close the Isaac Sim window to exit.\n")

    round_idx = 0
    while app.is_running():
        env.reset()
        env.step(zero_action)
        spd1 = torch.linalg.vector_norm(inner.object.data.root_lin_vel_w, dim=-1)
        launched = (spd1 > 2.0).nonzero(as_tuple=False).squeeze(-1).tolist()
        flag = f"  <-- LAUNCH envs {launched}" if launched else ""
        print(
            f"[reset-viz] round {round_idx:3d}  spd1 mean={spd1.mean():.2f}  "
            f"max={spd1.max():.2f} m/s (env {int(spd1.argmax())}){flag}"
        )
        for _ in range(args.reset_every - 1):
            if not app.is_running():
                break
            env.step(zero_action)
        round_idx += 1
        if args.rounds > 0 and round_idx >= args.rounds:
            break

    env.close()
    app.close()


if __name__ == "__main__":
    main()
