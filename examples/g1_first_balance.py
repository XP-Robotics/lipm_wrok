#!/usr/bin/env python3
"""
Very first balancing controller for the Unitree G1 MuJoCo model.

What this does:
- loads the robot in a standing keyframe
- keeps a nominal standing posture with position actuators
- adds torso pitch/roll feedback using hips and ankles

What this does NOT do yet:
- CoM/ZMP control
- single support
- stepping or walking
- push recovery

Run:
    python3 examples/g1_first_balance.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:
    import mujoco
    import mujoco.viewer
except ImportError as exc:
    raise SystemExit(
        "This example needs the `mujoco` Python package.\n"
        "Install it first, then run this file again."
    ) from exc


MODEL_PATH = "unitree_g1/scene.xml"


@dataclass
class Gains:
    torso_pitch_kp: float = 100.0
    torso_pitch_kd: float = 0.5
    torso_roll_kp: float = 100.0
    torso_roll_kd: float = 0.5

    ankle_pitch_gain: float = 0.1
    hip_pitch_gain: float = 0.35
    ankle_roll_gain: float = 0.1
    hip_roll_gain: float = 0.30


def rotation_matrix_to_roll_pitch_yaw(xmat: np.ndarray) -> tuple[float, float, float]:
    """Convert MuJoCo's flattened 3x3 rotation matrix to roll, pitch, yaw."""
    r = xmat.reshape(3, 3)
    pitch = math.asin(-np.clip(r[2, 0], -1.0, 1.0))
    roll = math.atan2(r[2, 1], r[2, 2])
    yaw = math.atan2(r[1, 0], r[0, 0])
    return roll, pitch, yaw


def actuator_id(model: mujoco.MjModel, name: str) -> int:
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if actuator_id < 0:
        raise KeyError(f"Actuator not found: {name}")
    return actuator_id


def keyframe_id(model: mujoco.MjModel, name: str) -> int:
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, name)
    if key_id < 0:
        raise KeyError(f"Keyframe not found: {name}")
    return key_id


def body_id(model: mujoco.MjModel, name: str) -> int:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise KeyError(f"Body not found: {name}")
    return bid


def sensor_id(model: mujoco.MjModel, name: str) -> int:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sid < 0:
        raise KeyError(f"Sensor not found: {name}")
    return sid


def sensor_vector(model: mujoco.MjModel, data: mujoco.MjData, sensor_name: str) -> np.ndarray:
    sid = sensor_id(model, sensor_name)
    start = model.sensor_adr[sid]
    dim = model.sensor_dim[sid]
    return np.array(data.sensordata[start : start + dim], copy=True)


def build_nominal_ctrl(model: mujoco.MjModel) -> np.ndarray:
    key_id = keyframe_id(model, "stand")
    ctrl = np.array(model.key_ctrl[key_id], copy=True)
    if ctrl.shape[0] != model.nu:
        raise RuntimeError("Unexpected actuator count in stand keyframe.")
    return ctrl


def apply_balance_feedback(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    nominal_ctrl: np.ndarray,
    gains: Gains,
    torso_body_id: int,
    act: dict[str, int],
) -> None:
    ctrl = nominal_ctrl.copy()

    torso_xmat = data.xmat[torso_body_id]
    torso_roll, torso_pitch, _ = rotation_matrix_to_roll_pitch_yaw(torso_xmat)

    torso_gyro = sensor_vector(model, data, "imu-torso-angular-velocity")
    roll_rate = torso_gyro[0]
    pitch_rate = torso_gyro[1]

    pitch_cmd = -(gains.torso_pitch_kp * torso_pitch + gains.torso_pitch_kd * pitch_rate)
    roll_cmd = -(gains.torso_roll_kp * torso_roll + gains.torso_roll_kd * roll_rate)

    ctrl[act["left_ankle_pitch_joint"]] += gains.ankle_pitch_gain * pitch_cmd
    ctrl[act["right_ankle_pitch_joint"]] += gains.ankle_pitch_gain * pitch_cmd
    ctrl[act["left_hip_pitch_joint"]] -= gains.hip_pitch_gain * pitch_cmd
    ctrl[act["right_hip_pitch_joint"]] -= gains.hip_pitch_gain * pitch_cmd

    ctrl[act["left_ankle_roll_joint"]] += gains.ankle_roll_gain * roll_cmd
    ctrl[act["right_ankle_roll_joint"]] += gains.ankle_roll_gain * roll_cmd
    ctrl[act["left_hip_roll_joint"]] -= gains.hip_roll_gain * roll_cmd
    ctrl[act["right_hip_roll_joint"]] -= gains.hip_roll_gain * roll_cmd

    data.ctrl[:] = np.clip(ctrl, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])


def main() -> None:
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    stand_key_id = keyframe_id(model, "stand")
    mujoco.mj_resetDataKeyframe(model, data, stand_key_id)
    mujoco.mj_forward(model, data)

    gains = Gains()
    nominal_ctrl = build_nominal_ctrl(model)
    torso_body_id = body_id(model, "torso_link")

    act = {
        "left_hip_pitch_joint": actuator_id(model, "left_hip_pitch_joint"),
        "left_hip_roll_joint": actuator_id(model, "left_hip_roll_joint"),
        "left_ankle_pitch_joint": actuator_id(model, "left_ankle_pitch_joint"),
        "left_ankle_roll_joint": actuator_id(model, "left_ankle_roll_joint"),
        "right_hip_pitch_joint": actuator_id(model, "right_hip_pitch_joint"),
        "right_hip_roll_joint": actuator_id(model, "right_hip_roll_joint"),
        "right_ankle_pitch_joint": actuator_id(model, "right_ankle_pitch_joint"),
        "right_ankle_roll_joint": actuator_id(model, "right_ankle_roll_joint"),
    }

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()
        while viewer.is_running():
            apply_balance_feedback(model, data, nominal_ctrl, gains, torso_body_id, act)
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
