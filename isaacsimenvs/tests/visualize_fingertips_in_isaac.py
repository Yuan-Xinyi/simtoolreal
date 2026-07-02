"""Visualize SimToolReal fingertip points in an Isaac Sim window.

Blue spheres follow the reward/observation fingertip points:
    DP body pose + FINGERTIP_OFFSET

Red spheres follow the URDF ``left_*_fingertip`` bodies when Isaac Lab exposes
them as articulation bodies. This makes it easy to compare the code-defined
reward point against the URDF fingertip frame.

Example:
    .venv_isaacsim/bin/python isaacsimenvs/tests/visualize_fingertips_in_isaac.py
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=int, default=0)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--num_assets_per_type", type=int, default=2)
    parser.add_argument("--radius", type=float, default=0.008)
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help=(
            "Number of policy steps to run. 0 means keep running while the app is open. "
            "Ignored by --freeze-reset."
        ),
    )
    parser.add_argument(
        "--freeze-reset",
        action="store_true",
        help="Show the reset hand pose only. The simulation renders but does not step physics.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app_launcher = AppLauncher(args)
    app = app_launcher.app

    import gymnasium as gym
    import omni.usd
    import torch
    from pxr import Gf, UsdGeom

    import isaacsimenvs  # noqa: F401
    from isaaclab.utils.math import quat_apply
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg import SimToolRealEnvCfg
    from isaacsimenvs.tasks.simtoolreal.utils.obs_utils import FINGERTIP_OFFSET

    cfg = SimToolRealEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.assets.num_assets_per_type = args.num_assets_per_type

    env = gym.make("Isaacsimenvs-SimToolReal-Direct-v0", cfg=cfg)
    inner = env.unwrapped
    env.reset()

    env_id = int(args.env_id)
    if env_id < 0 or env_id >= inner.num_envs:
        raise ValueError(f"--env_id must be in [0, {inner.num_envs - 1}], got {env_id}")

    stage = omni.usd.get_context().get_stage()
    root_path = f"/World/FingertipDebug/env_{env_id}"
    UsdGeom.Xform.Define(stage, root_path)

    finger_names = ("index", "middle", "ring", "thumb", "pinky")
    dp_ids = list(inner._fingertip_body_ids)
    dp_body_names = [inner.robot.data.body_names[i] for i in dp_ids]

    try:
        urdf_tip_ids = list(inner.robot.find_bodies("left_.*_fingertip")[0])
    except Exception:  # noqa: BLE001
        urdf_tip_ids = []
    urdf_tip_names = [inner.robot.data.body_names[i] for i in urdf_tip_ids]

    def make_sphere(path: str, color: tuple[float, float, float], radius: float):
        sphere = UsdGeom.Sphere.Define(stage, path)
        sphere.CreateRadiusAttr(radius)
        sphere.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        xform = UsdGeom.Xformable(sphere.GetPrim())
        translate_op = xform.AddTranslateOp()
        return translate_op

    blue_ops = []
    red_ops = []
    for i, name in enumerate(dp_body_names):
        blue_ops.append(
            make_sphere(
                f"{root_path}/reward_tip_{i}_{name}",
                (0.05, 0.25, 1.0),
                args.radius,
            )
        )
    for i, name in enumerate(urdf_tip_names):
        red_ops.append(
            make_sphere(
                f"{root_path}/urdf_tip_{i}_{name}",
                (1.0, 0.05, 0.04),
                args.radius * 0.75,
            )
        )

    print("\n[fingertip-viz] Blue spheres: reward/obs fingertip points")
    for i, name in enumerate(dp_body_names):
        print(f"  blue[{i}] {name} + local offset {FINGERTIP_OFFSET}")
    if urdf_tip_names:
        print("[fingertip-viz] Red spheres: URDF fingertip bodies")
        for i, name in enumerate(urdf_tip_names):
            print(f"  red [{i}] {name}")
    else:
        print("[fingertip-viz] Red spheres skipped: no left_*_fingertip bodies exposed.")
    if args.freeze_reset:
        print("[fingertip-viz] Freeze-reset mode: showing the reset hand pose without stepping physics.")
    print("[fingertip-viz] Close the Isaac Sim window to exit.\n")

    offset = torch.tensor(FINGERTIP_OFFSET, device=inner.device, dtype=torch.float32)
    zero_action = torch.zeros(
        (inner.num_envs, inner.cfg.action_space), device=inner.device, dtype=torch.float32
    )

    def update_markers() -> None:
        dp_state = inner.robot.data.body_state_w[env_id, dp_ids, :]
        dp_pos = dp_state[:, 0:3]
        dp_quat = dp_state[:, 3:7]
        reward_pos = dp_pos + quat_apply(dp_quat, offset.expand(len(dp_ids), -1))
        for op, pos in zip(blue_ops, reward_pos.detach().cpu().tolist()):
            op.Set(Gf.Vec3d(*pos))

        if urdf_tip_ids:
            urdf_pos = inner.robot.data.body_state_w[env_id, urdf_tip_ids, 0:3]
            for op, pos in zip(red_ops, urdf_pos.detach().cpu().tolist()):
                op.Set(Gf.Vec3d(*pos))

    count = 0
    while app.is_running():
        update_markers()
        if args.freeze_reset:
            app.update()
            continue
        env.step(zero_action)
        count += 1
        if args.steps > 0 and count >= args.steps:
            break

    env.close()
    app.close()


if __name__ == "__main__":
    main()
