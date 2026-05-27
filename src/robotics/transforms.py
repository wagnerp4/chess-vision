from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

try:
    import pinocchio as pin
except ImportError as exc:
    import sys

    _msg = (
        "Pinocchio is required for vision.src.robotics.transforms. "
        "On Linux you can install the PyPI package named pin. "
        "On macOS pip wheels are not published upstream. "
        "Use conda-forge instead: conda install pinocchio -c conda-forge."
    )
    if sys.platform.startswith("win"):
        _msg += " On Windows use conda-forge or build from source."
    raise ImportError(_msg) from exc


def se3_from_homogeneous(T: np.ndarray) -> pin.SE3:
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4):
        raise ValueError("T must be 4x4 homogeneous")
    R = T[:3, :3]
    t = T[:3, 3]
    return pin.SE3(R, t)


def homogeneous_from_se3(M: pin.SE3) -> np.ndarray:
    H = np.eye(4, dtype=float)
    H[:3, :3] = np.asarray(M.rotation, dtype=float)
    H[:3, 3] = np.asarray(M.translation, dtype=float).reshape(3)
    return H


def se3_from_rotation_translation(R: np.ndarray, t: np.ndarray) -> pin.SE3:
    R = np.asarray(R, dtype=float).reshape(3, 3)
    t = np.asarray(t, dtype=float).reshape(3)
    return pin.SE3(R, t)


def points_camera_to_tcp(T_tcp_cam: pin.SE3, p_cam: np.ndarray) -> np.ndarray:
    pts = np.asarray(p_cam, dtype=float)
    if pts.shape[-1] != 3:
        raise ValueError("last dimension must be 3")
    flat = pts.reshape(-1, 3)
    R = np.asarray(T_tcp_cam.rotation, dtype=float)
    t = np.asarray(T_tcp_cam.translation, dtype=float).reshape(1, 3)
    out = flat @ R.T + t
    return out.reshape(pts.shape)


def world_tcp_from_world_camera(T_world_cam: pin.SE3, T_tcp_cam: pin.SE3) -> pin.SE3:
    return T_world_cam * T_tcp_cam.inverse()


def load_model_and_data(
    urdf_path: str | Path,
    package_dirs: Sequence[str | Path] | None = None,
) -> tuple[pin.Model, pin.Data]:
    urdf_path = Path(urdf_path).expanduser().resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(str(urdf_path))
    if package_dirs is None:
        dirs = [str(urdf_path.parent)]
    else:
        dirs = [str(Path(p).expanduser().resolve()) for p in package_dirs]
    # TODO: If mesh loading fails, add every parent directory that contains package:// roots.
    model = pin.buildModelFromUrdf(str(urdf_path), dirs)
    data = model.createData()
    return model, data


def frame_placement_in_universe(
    model: pin.Model,
    data: pin.Data,
    q: np.ndarray,
    frame_name: str,
) -> pin.SE3:
    q = np.asarray(q, dtype=float).reshape(model.nq)
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    # TODO: Use the URDF frame name that matches your calibrated TCP or tool flange.
    fid = model.getFrameId(frame_name)
    return data.oMf[fid]


def verify_camera_tcp_helpers() -> None:
    R_tcp_cam = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    t_tcp_cam = np.array([0.1, -0.2, 0.35], dtype=float)
    T_tcp_cam = se3_from_rotation_translation(R_tcp_cam, t_tcp_cam)

    H = homogeneous_from_se3(T_tcp_cam)
    T_round = se3_from_homogeneous(H)
    if not np.allclose(
        homogeneous_from_se3(T_tcp_cam),
        homogeneous_from_se3(T_round),
        rtol=0.0,
        atol=1e-10,
    ):
        raise AssertionError("homogeneous SE3 round-trip mismatch")

    p_cam = np.array(
        [
            [1.0, 0.0, -0.5],
            [0.0, 2.0, 0.25],
            [-0.3, 0.1, 0.0],
        ],
        dtype=float,
    )
    p_tcp = points_camera_to_tcp(T_tcp_cam, p_cam)
    for i in range(p_cam.shape[0]):
        ref = np.asarray(T_tcp_cam.act(p_cam[i]), dtype=float).reshape(3)
        if not np.allclose(ref, p_tcp[i], rtol=0.0, atol=1e-10):
            raise AssertionError("points_camera_to_tcp disagrees with pin.SE3.act")

    p_cam_back = points_camera_to_tcp(T_tcp_cam.inverse(), p_tcp)
    if not np.allclose(p_cam, p_cam_back, rtol=0.0, atol=1e-10):
        raise AssertionError("camera tcp inverse round-trip mismatch")

    p_cam_batched = p_cam.reshape(1, 3, 3)
    p_tcp_b = points_camera_to_tcp(T_tcp_cam, p_cam_batched)
    p_back_b = points_camera_to_tcp(T_tcp_cam.inverse(), p_tcp_b)
    if not np.allclose(p_cam_batched, p_back_b, rtol=0.0, atol=1e-10):
        raise AssertionError("batched camera tcp round-trip mismatch")

    R_world_cam = np.array(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=float,
    )
    t_world_cam = np.array([-0.05, 0.4, 0.6], dtype=float)
    T_world_cam = se3_from_rotation_translation(R_world_cam, t_world_cam)
    T_world_tcp = world_tcp_from_world_camera(T_world_cam, T_tcp_cam)
    p_world_a = np.asarray(T_world_cam.act(p_cam[0]), dtype=float).reshape(3)
    p_tcp0 = np.asarray(T_tcp_cam.act(p_cam[0]), dtype=float).reshape(3)
    p_world_b = np.asarray(T_world_tcp.act(p_tcp0), dtype=float).reshape(3)
    if not np.allclose(p_world_a, p_world_b, rtol=0.0, atol=1e-10):
        raise AssertionError("world_tcp_from_world_camera composition mismatch")


def points_camera_to_universe_via_tcp(
    model: pin.Model,
    data: pin.Data,
    q: np.ndarray,
    tcp_frame_name: str,
    T_tcp_cam: pin.SE3,
    p_cam: np.ndarray,
) -> np.ndarray:
    oMtcp = frame_placement_in_universe(model, data, q, tcp_frame_name)
    p_tcp = points_camera_to_tcp(T_tcp_cam, p_cam)
    pts = np.asarray(p_tcp, dtype=float)
    if pts.shape[-1] != 3:
        raise ValueError("last dimension must be 3")
    flat = pts.reshape(-1, 3)
    R = np.asarray(oMtcp.rotation, dtype=float)
    t = np.asarray(oMtcp.translation, dtype=float).reshape(1, 3)
    out = flat @ R.T + t
    return out.reshape(pts.shape)


__all__ = [
    "homogeneous_from_se3",
    "frame_placement_in_universe",
    "load_model_and_data",
    "points_camera_to_tcp",
    "points_camera_to_universe_via_tcp",
    "se3_from_homogeneous",
    "se3_from_rotation_translation",
    "verify_camera_tcp_helpers",
    "world_tcp_from_world_camera",
]


if __name__ == "__main__":
    verify_camera_tcp_helpers()
    print("verify_camera_tcp_helpers: ok")
