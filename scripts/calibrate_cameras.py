#!/usr/bin/env python3
"""Calibrate the DROID's camera views against the arm base, in one command per camera.

The same views, boards, jog and solve as openyam_deploy's scripts/calibrate_cameras.py, on the DROID's Franka (through
the NUC's control server) and its ZED cameras (README.md, Calibrate).

A camera's VIEW is the image the policy sees: its raw frame reprojected to a pinhole image of a given size and
intrinsics, turned by a given rotation about the camera's optical centre. This script calibrates the views, not the raw
cameras, treating the reprojection as a black box: the board is detected in the raw frame, its markers are mapped into
the view exactly as the reprojection maps pixels, only markers wholly inside the view are kept, and the view is
calibrated as a camera of its own.

It reads the arm's joints and the cameras through droid.Droid (the NUC's control server and the ZED SDK), builds FK
from the config's URDF (configs/franka_droid.urdf, with the node graph's TCP added as the body "gripper/tcp", the
fingertips) and, while the arm moves through varied poses (--jog drives it), records every camera's raw frame together
with the FK poses of the gripper and of each camera's parent body. Then per camera it calibrates the view in one fit
over every frame: its intrinsics, its pose, the board's pose and, per frame, a small correction of the arm's pose, held
near zero by the arm's accuracy, which the fit measures itself. So each image fixes its own frame's pose, the arm's pose
errors can't leak into the intrinsics, and the arm's motion still pins what the images alone can't. It cross-checks the
result and writes it. A capture redoes that solve at every frame that shows the board, prints its errors, and stops
once the solve's uncertainty is low enough.

Each camera declares in the config its ChArUco board and its mount:
  * "mount": "gripper" -> board on the GRIPPER, static camera -> eye-to-hand -> view-in-base
  * "mount": "fixed"   -> board FIXED in the scene, moving camera -> eye-in-hand -> view-in-link
Its raw lens and its view are completed as the server completes them (droid.complete): "raw" is the ZED's rectified
left image, a pinhole with the SDK's intrinsics, 1280x720; "view" is the config's {"size", "fov_deg"}, a centred pinhole
along the camera's own axes. The capture saves the completed entries (cameras.json in the work dir), so --reuse-capture
re-solves without the robot. The output (--config-out) is the input config with each solved camera's completed "raw"
and "view" and its "calibration": the view camera as calibrated, valid only for that recipe:
  * "K", "D": its intrinsics, in OpenCV's standard model (D = k1, k2, p1, p2, k3; only k1 and k2 are fitted);
  * "optical_in_parent": its pose, the optical frame (OpenCV axes) in the camera's parent body;
  * "board_in_ee" (gripper board, in "ee_body") or "board_in_base" (fixed board): where the board was;
  * the fit numbers: "frames", "rms_px", "arm_error_deg_mm" (the arm's pose error per frame, as the fit measured
    it), "uncertainty_px" (how far refits, each without a share of the frames, disagree on where points across the
    view land, 95th percentile), "cross_check"; and "calibrated", when the capture was made.
serve.py and visualize.py take that config: a camera with its own "calibration" uses it (its ZED must report the same
lens), the others the DROID framework's calibration files, as before.

The jog moves the Franka on its own: someone stands at the e-stop. Nothing on the Franka refuses a self-collision, so
every pose the jog sends the arm to, and the move there, is checked from FK first
(_clearance): each joint within JOG_HALF_RANGE_DEG x --amp of home (the arm's initial pose: a camera's jog.start_deg,
else the serve start pose; or where the first steps found the board); the fingertips, the camera links and (once the
jog has modelled it) the gripper board at least JOG_TABLE_CLEARANCE_M above the table (the config's "table_z_m",
measured: scripts/arm.py where); and the fingertips and the board at least JOG_KEEPOUT's distances from the arm's base
column, upper arm and forearm.

Every camera with a "board" is calibrated by default; --cameras narrows it (README.md, Calibrate: one command per
camera). `calib/make_charuco.py` prints the exact board a camera's config declares.

  .venv/bin/python scripts/calibrate_cameras.py --jog --cameras wrist \\
      --config configs/droid_cameras.json --config-out out/droid_cameras_calibrated.json \\
      --work-dir out/calib-wrist     # boards and mounts come from the config

Add --reuse-capture to re-solve an existing --work-dir without touching the robot. The capture keeps the raw frames,
so a changed view needs no new capture.
"""
import argparse
import importlib.util
import json
import math
import os
import signal
import threading
import time
from collections import Counter
from pathlib import Path

for _blas in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):   # the solves' matrices are tiny: BLAS threads only add
    os.environ.setdefault(_blas, '1')                      # overhead (3.5x slower on 8 cores); set before numpy loads
import numpy as np

from droid_deploy import REPO
from droid_deploy.droid import JOINT_MAX_AGE_S, NotReady, RobotError
from droid_deploy.geometry import inv_se3, rt_to_se3
from droid_deploy.kinematics import build_model, fk_poses
from droid_deploy.motion import SLEW_RAD_S as JOG_SLEW_RAD_S, Stopped, hold as _hold, move as _move
from droid_deploy.views import camera_models as _models, to_view as _to_view, view_maps as _view_maps

FRAME_WAIT_S = 2.0       # after the arm settles, how long to wait for a new frame from every camera


def _have(mod):
    try:
        return importlib.util.find_spec(mod) is not None
    except ModuleNotFoundError:
        return False


# ---- SE(3) ------------------------------------------------------------------------
def _mean_pose(poses):
    """The average of rigid transforms: mean translation, rotation = the nearest rotation to the mean matrix."""
    u, _, vt = np.linalg.svd(np.mean([T[:3, :3] for T in poses], axis=0))
    R = u @ vt
    if np.linalg.det(R) < 0:
        R = u @ np.diag([1.0, 1.0, -1.0]) @ vt
    return rt_to_se3(R, np.mean([T[:3, 3] for T in poses], axis=0))


# ---- scripted auto-jog for coverage -----------------------------------------------
JOG_SETTLE_S = 0.4                       # settle before grabbing a frame
JOG_MARGIN_RAD = math.radians(5.0)       # stay this far inside each joint's mechanical limit
JOG_HALF_RANGE_DEG = (35., 20., 20., 20., 45., 45., 60.)   # per arm-joint half-range around the start (the Franka's
                                          # seven: the shoulder, upper arm and elbow kept tight); wrist-heavy
JOG_BIAS_FRAC = 0.5                       # present-to-waist: fraction of waypoints retracted toward the base camera
JOG_MAX_TURN_DEG = 25.0                   # a camera on the arm (fixed board) turns at most this far from the
                                          # look-down pose the jog samples around, so the board below stays in view
JOG_MAX_DOWN_OFF_DEG = 30.0               # a lens capture of a camera on the arm (no view): refuse the jog when the
                                          # gripper can't point within this of straight down
JOG_SEARCH_STEP_DEG = 10.0                # a camera on the arm: the jog's first steps tilt its view up (toward its
                                          # image top) this far at a time until it sees its board ...
JOG_SEARCH_MIN_BELOW_DEG = 20.0           # ... while its view still looks at least this far below the horizon ...
JOG_SEARCH_TOL_DEG = 3.0                  # ... and the arm can tilt it to within this of each step
JOG_SEARCH_JOINTS = ((1, 'joint 2, the shoulder'), (3, 'joint 4, the elbow'), (5, 'joint 6, the wrist'))
                                          # ... each step turned by one of these arm joints alone (the shoulder
                                          # first), else the gripper held
JOG_REACH_STEP_M = 0.04                   # a board on the gripper: the jog's first steps move the gripper this far
                                          # up and forward (half each) at a time, its orientation held, until the
                                          # views see the board ...
JOG_REACH_MAX_M = 0.40                    # ... at most this far in all
JOG_MAX_SKIPS = 30                        # the jog stops after this many waypoints in a row fail the clearance checks
JOG_PRESENT_RETRACT_DEG = (12., 35.)      # present-to-waist: retract offset to bring the gripper board nearer the base cam
JOG_TABLE_CLEARANCE_M = 0.10              # keep the fingertips, the camera links and the gripper board this far above
                                          # the table, m
JOG_FLOOR_Z = None                        # the config's "table_z_m" (the table, in the base) + JOG_TABLE_CLEARANCE_M:
                                          # set by capture()
JOG_KEEPOUT = (('fr3_link0', 'fr3_link2', 1.0, 0.15),   # the fingertips and the gripper board stay this far (m) from
               ('fr3_link2', 'fr3_link4', 1.0, 0.15),   # the segment between these bodies' origins, cut at the fraction
               ('fr3_link4', 'fr3_link5', 0.7, 0.12))   # toward the second: the base column, the upper arm, and the
                                          # forearm without its last 30 % (where the hand hangs). Nothing on the Franka
                                          # refuses a self-collision; at the DROID start pose the fingertips are
                                          # 0.29-0.32 m from all three
JOG_BASE_MIN = 0.28                       # present-to-waist: never bring the tcp closer than this to the base (avoid folding in)
JOG_STOP_FRAMES = 5                       # a view jog ends once each view's uncertainty has been <= --uncertainty-px at
                                          # this many of its board frames in a row
JOG_EXPLORE_FRAC = 0.4                    # until it can aim at the board, the jog stays within this fraction of each
                                          # joint's range around home, where the board was in view


def _arm_limits(model, command_joints):
    import mujoco
    out = []
    for jn in command_joints:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid >= 0 and model.jnt_limited[jid]:
            out.append((float(model.jnt_range[jid][0]), float(model.jnt_range[jid][1])))
        else:
            out.append((-math.inf, math.inf))
    return out


def _tcp_range_fn(model, data, bodies, ref, ee):
    """Distance from the tcp to the base (~ the static waist camera): decreasing it retracts the
    gripper toward that camera, so its board fills more of the waist frame (better PnP)."""
    def f(jd):
        poses = fk_poses(model, data, jd, bodies)
        return float(np.linalg.norm(poses[ee][:3, 3] - poses[ref][:3, 3]))
    return f


def _dominant_joint(model, data, home_jd, cj, n_arm, scalar_fn):
    """The arm joint whose motion most changes scalar_fn, and the sign that DECREASES it."""
    z0 = scalar_fn(home_jd)
    delta = math.radians(10.0)
    best_j, best_grad = 0, 0.0
    for i in range(n_arm):
        jd = dict(home_jd); jd[cj[i]] = home_jd[cj[i]] + delta
        grad = (scalar_fn(jd) - z0) / delta
        if abs(grad) > abs(best_grad):
            best_j, best_grad = i, grad
    return best_j, (-1.0 if best_grad > 0 else 1.0)


def _view_axes(cam, entry):
    """The wrist camera's view axes (OpenCV: x right, y down, z forward) in its parent body, from its calibration: its
    own when the config has one (a calibrated view), else the DROID framework's that droid.py loads (the camera's axes,
    turned by the view's R). The tilt search needs them to know which way is up in the view; roughly right is enough."""
    calibration = entry.get('calibration') or {}
    if calibration.get('optical_in_parent') is None:
        raise SystemExit(f'{cam}: no calibration to start from (neither its own nor in the DROID calibration files): the '
                         'jog\'s first steps need to know roughly where its view looks')
    axes = np.asarray(calibration['optical_in_parent'], float)[:3, :3]
    if 'frames' not in calibration:                      # the DROID framework's: the raw camera's axes
        axes = axes @ np.asarray((entry.get('view') or {}).get('R', np.eye(3)), float)
    return axes


def _point_view(model, data, cj, n_arm, start_jd, bodies, ref, ee, parent, axis, want, low_bodies, lims):
    """Arm joints that point ``axis`` (a unit vector in the ``parent`` body: where the wrist camera looks, or the
    gripper's approach) along ``want`` (a unit vector in the base), with the tcp near where it is at ``start_jd`` and
    every low body above the table. Returns (the arm joints, how far the axis ends off ``want`` in deg, how far the
    gripper's approach is off vertical in deg, how far the tcp moved in m)."""
    from scipy.optimize import minimize
    want = np.asarray(want, float)

    def state(q):
        poses = fk_poses(model, data, {**start_jd, **{cj[k]: float(q[k]) for k in range(n_arm)}}, bodies)
        w2b = inv_se3(poses[ref])
        tcp, p = w2b @ poses[ee], w2b @ poses[parent]
        approach = (tcp[:3, 3] - p[:3, 3]) / np.linalg.norm(tcp[:3, 3] - p[:3, 3])
        return tcp[:3, 3], p[:3, :3] @ axis, approach, min(float((w2b @ poses[b])[2, 3]) for b in low_bodies)
    q0 = np.array([start_jd[cj[k]] for k in range(n_arm)], float)
    p0 = state(q0)[0]

    def cost(q):                    # the aim first: ~3 deg off costs as much as a 15 cm tcp shift
        tcp, a, _, low = state(q)
        return ((np.linalg.norm(a - want) / 0.05) ** 2 + (np.linalg.norm(tcp - p0) / 0.15) ** 2
                + (max(0.0, JOG_FLOOR_Z - low) / 0.01) ** 2)
    bounds = [(lo + JOG_MARGIN_RAD if math.isfinite(lo) else None, hi - JOG_MARGIN_RAD if math.isfinite(hi) else None)
              for lo, hi in lims[:n_arm]]
    q = minimize(cost, q0, method='L-BFGS-B', bounds=bounds).x
    tcp, a, approach, _ = state(q)
    angle = lambda u, v: math.degrees(math.acos(float(np.clip(u @ v, -1.0, 1.0))))
    return q, angle(a, want), angle(approach, np.array([0.0, 0.0, -1.0])), float(np.linalg.norm(tcp - p0))


def _reach(model, data, cj, n_arm, start_jd, bodies, ref, ee, target, low_bodies, lims):
    """Arm joints that put the tcp at ``target`` (4x4 in the base), from ``start_jd``, every low body above the table.
    Returns (the arm joints, how far the tcp ends from the target in m, how far it ends turned from it in deg)."""
    from scipy.optimize import minimize

    def state(q):
        poses = fk_poses(model, data, {**start_jd, **{cj[k]: float(q[k]) for k in range(n_arm)}}, bodies)
        w2b = inv_se3(poses[ref])
        tcp = w2b @ poses[ee]
        return tcp, min(float((w2b @ poses[b])[2, 3]) for b in low_bodies)

    def cost(q):                    # 1 mm off costs as much as 0.5 deg turned
        tcp, low = state(q)
        return ((np.linalg.norm(tcp[:3, 3] - target[:3, 3]) / 0.001) ** 2
                + (np.linalg.norm(_log_so3(target[:3, :3].T @ tcp[:3, :3])) / math.radians(0.5)) ** 2
                + (max(0.0, JOG_FLOOR_Z - low) / 0.001) ** 2)
    q0 = np.array([start_jd[cj[k]] for k in range(n_arm)], float)
    bounds = [(lo + JOG_MARGIN_RAD if math.isfinite(lo) else None, hi - JOG_MARGIN_RAD if math.isfinite(hi) else None)
              for lo, hi in lims[:n_arm]]
    q = minimize(cost, q0, method='L-BFGS-B', bounds=bounds).x
    tcp, _ = state(q)
    return (q, float(np.linalg.norm(tcp[:3, 3] - target[:3, 3])),
            math.degrees(float(np.linalg.norm(_log_so3(target[:3, :3].T @ tcp[:3, :3])))))


def _turn_limit(model, data, cj, joints, bodies, ref, body, max_deg):
    """(``body``, its rotation in the base at ``joints``, ``max_deg`` in rad): _make_safe keeps ``body`` turned at most
    that far from there."""
    poses = fk_poses(model, data, {cj[k]: float(joints[k]) for k in range(len(cj))}, bodies)
    return body, (inv_se3(poses[ref]) @ poses[body])[:3, :3], math.radians(max_deg)


_BOARD_CORNERS = []      # the gripper board's corners in ee_body, from the jog's board model once it has one
                         # (_note_board): the clearance checks then keep the board off the table and the arm too


def _note_board(planner):
    """The gripper board's corners in ee_body, from ``planner``'s full fit (a static camera's: the board on the
    gripper), into _BOARD_CORNERS."""
    B = planner._unpack(planner.fits[0])[1]
    _BOARD_CORNERS[:] = [(B @ c)[:3] for c in planner.outline]


def _segment_distance(p, a, b):
    """The distance from the point ``p`` to the segment from ``a`` to ``b``."""
    ab = b - a
    s = float(np.clip((p - a) @ ab / max(float(ab @ ab), 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + s * ab)))


def _clearance(poses, ref, ee, low_bodies, z_floor):
    """What makes the arm's pose unsafe, or None (``poses``: its fk_poses, with ``ee`` and JOG_KEEPOUT's bodies): a low
    body, or the gripper board's corners once modelled (_BOARD_CORNERS), below ``z_floor`` in the base (the table +
    JOG_TABLE_CLEARANCE_M; None: not checked), or the fingertips (``ee``) or the board's corners nearer the arm's own
    links than JOG_KEEPOUT. Nothing on the Franka refuses a self-collision, so the jog checks every pose it sends the
    arm to, and the move there (_path_blocked, _make_safe)."""
    w2b = inv_se3(poses[ref])
    tip = w2b @ poses[ee]
    board = [tip[:3, :3] @ c + tip[:3, 3] for c in _BOARD_CORNERS]
    if z_floor is not None:
        for b in low_bodies:
            if float((w2b @ poses[b])[2, 3]) < z_floor:
                return f'{b} would come within {100 * JOG_TABLE_CLEARANCE_M:.0f} cm of the table'
        if any(float(p[2]) < z_floor for p in board):
            return f'the gripper board would come within {100 * JOG_TABLE_CLEARANCE_M:.0f} cm of the table'
    for a, b, frac, near in JOG_KEEPOUT:
        pa = (w2b @ poses[a])[:3, 3]
        pb = pa + frac * ((w2b @ poses[b])[:3, 3] - pa)
        for what, p in [('the fingertips', tip[:3, 3])] + [('the gripper board', q) for q in board]:
            if _segment_distance(p, pa, pb) < near:
                return f'{what} would come within {100 * near:.0f} cm of the arm ({a} to {b})'
    return None


def _path_blocked(model, data, cj, a, b, bodies, ref, ee, low_bodies, z_floor, n=20):
    """What makes the joint-space move from ``a`` to ``b`` unsafe somewhere along it (_clearance at ``n`` points), or
    None."""
    for s in np.linspace(0.0, 1.0, n + 1)[1:]:
        poses = fk_poses(model, data, {cj[k]: float(a[k] + (b[k] - a[k]) * s) for k in range(len(cj))}, bodies)
        why = _clearance(poses, ref, ee, low_bodies, z_floor)
        if why:
            return why
    return None


def _make_safe(model, data, cj, target, home, bodies, ref, ee, low_bodies, z_floor, base_min=None, turn=None):
    """Scale a waypoint back toward home until it is safe: it passes the clearance checks (_clearance: the table and
    the arm's own links), (if base_min set) the tcp stays >= base_min from the base so a retract can't fold into the
    base, and (with turn = (body, its rotation in the base frame at home, max angle)) that body turns at most max angle
    from home: a camera fixed on it then looks at most that far from where it looked at home, however it is mounted."""
    for s in (1.0, 0.8, 0.6, 0.45, 0.3, 0.15, 0.0):
        cand = home + (target - home) * s
        jd = {cj[k]: float(cand[k]) for k in range(len(cj))}
        poses = fk_poses(model, data, jd, bodies)
        w2b = inv_se3(poses[ref])
        z_ok = _clearance(poses, ref, ee, low_bodies, z_floor) is None
        d_ok = base_min is None or float(np.linalg.norm(poses[ee][:3, 3] - poses[ref][:3, 3])) >= base_min
        t_ok = turn is None or float(np.linalg.norm(_log_so3(turn[1].T @ (w2b @ poses[turn[0]])[:3, :3]))) <= turn[2]
        if z_ok and d_ok and t_ok:
            return cand
    return home.copy()


def _write_png(rgb, path):
    import cv2
    cv2.imwrite(str(path), cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR))


def _write_mp4(cam, work, subdir='frames', suffix='', fps=4):
    """Assemble a per-camera PNG stream (work/<subdir>/<cam>/step_*.png) into a watchable MP4."""
    import cv2
    pngs = sorted((Path(work) / subdir / cam).glob('step_*.png'))
    if not pngs:
        return
    h, w = cv2.imread(str(pngs[0])).shape[:2]
    out = Path(work) / f'{cam}{suffix}.mp4'
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for p in pngs:
        vw.write(cv2.imread(str(p)))
    vw.release()
    print(f'wrote {out}  ({len(pngs)} frames @ {fps} fps)', flush=True)


def _save_frame(ros, model, data, cams, ref, ee, parents, bodies, work, idx, since=0.0, wait=0.0, annotate=None):
    """One frame of every camera, newer than ``since``, with the joints and the FK poses in the base frame, as
    step_<idx>.npz plus a PNG per camera, and with ``annotate`` (_live_annotator) its annotated view. Returns (what
    ``annotate`` returns, or {}; the poses as saved: {"ee_in_base", "parent__<camera>"}). NotReady if a camera or the
    joints have nothing fresh."""
    frames, joints = ros.grab(cams, since, wait)
    poses = fk_poses(model, data, joints, bodies)
    w2b = inv_se3(poses[ref])
    names = sorted(joints)
    saved = {'ee_in_base': (w2b @ poses[ee]).astype(np.float32),
             **{f'parent__{c}': (w2b @ poses[parents[c]]).astype(np.float32) for c in cams}}
    np.savez_compressed(
        Path(work) / f'step_{idx:04d}.npz', **saved,
        joint_names=np.array(names), joint_pos=np.array([joints[k] for k in names]),
        **{f'raw__{c}': frames[c] for c in cams})
    for c in cams:
        pdir = Path(work) / 'frames' / c
        pdir.mkdir(parents=True, exist_ok=True)
        _write_png(frames[c], pdir / f'step_{idx:04d}.png')
    return (annotate(idx, frames) if annotate else {}), saved


# ---- the jog aims at the board once it has seen it ----------------------------------------------------------------
PLAN_MIN_FRAMES = 4       # frames that show the board before the jog aims at it
PLAN_FOLDS = 6            # the model's uncertainty: refits, each without one of this many shares of the frames
PLAN_MIN_MARKERS = 8      # an aimed waypoint needs every refit to show this many whole markers inside the view
                          # (a camera's config "jog": {"aim_min_markers": n} sets its own),
PLAN_MARGIN_PX = 16       # at least this far in from its edges (view pixels),
PLAN_MIN_MARKER_PX = 5.0  # each at least this big (view pixels; smaller ones can go undetected in the raw frame),
PLAN_MAX_TILT_DEG = 60.0  # with the board facing the view within this,
PLAN_NEAR_M = 0.04        # near a frame that showed the board: within this of its distance from the view ...
PLAN_NEAR_DEG = 10.0      # ... and this of how it faced the view (its normal), the two scaled alike; and nearer
                          # such a frame than one that showed none of it though it was in view
PLAN_CANDIDATES = 80      # candidate waypoints drawn per aimed waypoint; the one least like the frames so far is used,
PLAN_NOVELTY = (20.0, 40.0)   # judged by how the board is turned (deg) and where it lands in the view (px), alike
PLAN_MAX_TURN_DEG = 45.0  # a camera on the arm: an aimed waypoint turns it at most this far from the look-down pose


class _BoardPlanner:
    """Where a camera's board is, learned during the jog from the frames that show it, and which waypoints will show it
    well. The model is the view's pose X and the board's pose B: for a camera on the arm (board fixed), X on the
    camera's link and B in the base; for a static camera (``on_ee``: the board on the gripper), X in the base and B on
    the gripper. In a frame the board sits in the view at inv(X) @ M @ B, with M from the FK of the frame's "move": the
    inverse of the camera link's pose, or the gripper's pose. The model is fitted to the centres of the markers the view
    saw (view pixels, the recipe's view K) from a hand-eye start, and refitted PLAN_FOLDS times, each without one share
    of the frames: with few frames the refits disagree, and a waypoint counts as showing the board only if every one of
    them says it does. It is also only trusted near what the view has shown: an aimed waypoint must put the board at
    a distance and facing near those of a frame that showed it, and nearer such a frame than one that missed it while
    it was in view (PLAN_NEAR_M, PLAN_NEAR_DEG), so the region it aims at grows from where the board was read and
    shrinks where it was not."""

    def __init__(self, view, board_spec, on_ee, min_markers=PLAN_MIN_MARKERS):
        self.K, self.size, self.on_ee, self.min_markers = view['K'], view['size'], on_ee, int(min_markers)
        _, obj = _board(board_spec)
        self.corners = np.stack([obj[m] for m in sorted(obj)])                 # (markers, 4, 3), board frame
        s = board_spec['square_len_mm'] / 1000.0
        self.centre = np.r_[board_spec['squares_x'] * s / 2, board_spec['squares_y'] * s / 2, 0.0, 1.0]
        w, h = board_spec['squares_x'] * s, board_spec['squares_y'] * s
        self.outline = np.array([[x, y, 0.0, 1.0] for x, y in ((0, 0), (w, 0), (w, h), (0, h))])   # the board's corners
        self.moves, self.obj, self.img, self.fits = [], [], [], []
        self.owner = np.zeros(0, int)                       # each marker centre's frame
        self.misses = []                                    # the moves of frames that showed none of the board

    def _inside(self, T):
        """Which markers are wholly inside the view, PLAN_MARGIN_PX in from its edges, with the board at T."""
        (w, h), m = self.size, PLAN_MARGIN_PX
        P, uv = self._project(T, self.corners)
        inside = ((P[..., 2] > 0.05) & (uv[..., 0] >= m) & (uv[..., 0] <= w - 1 - m)
                  & (uv[..., 1] >= m) & (uv[..., 1] <= h - 1 - m)).all(-1)
        return inside, uv

    def _facing(self, T):
        """How the board sits in the view at T: its centre's distance (m), its normal (the view's frame), and how far
        it is turned from facing the view (deg)."""
        c = (T @ self.centre)[:3]
        d = float(np.linalg.norm(c))
        return d, T[:3, 2], math.degrees(math.acos(float(np.clip(T[:3, 2] @ c / d, -1.0, 1.0))))

    def _facings(self):
        """The frames' distances and normals with the full fit: those that showed the board, and those that showed
        none of it while the fit puts it in view."""
        X, B = self._unpack(self.fits[0])
        seen = [self._facing(inv_se3(X) @ self._M(m) @ B) for m in self.moves]
        missed = []
        for m in self.misses:
            T = inv_se3(X) @ self._M(m) @ B
            if int(self._inside(T)[0].sum()) >= self.min_markers:
                missed.append(self._facing(T))
        self.seen_d, self.seen_n = np.array([f[0] for f in seen]), np.array([f[1] for f in seen])
        self.seen_tilt = np.array([f[2] for f in seen])
        self.missed_d = np.array([f[0] for f in missed])
        self.missed_n = np.array([f[1] for f in missed]).reshape(-1, 3)

    @staticmethod
    def _nearest(d, n, ds, ns):
        """How near (distance, normal) is the nearest of (ds, ns), in PLAN_NEAR_M / PLAN_NEAR_DEG (inf if none)."""
        if len(ds) == 0:
            return math.inf
        turned = np.degrees(np.arccos(np.clip(ns @ n, -1.0, 1.0)))
        return float(np.min(np.hypot((ds - d) / PLAN_NEAR_M, turned / PLAN_NEAR_DEG)))

    def miss(self, move):
        """A frame that showed none of the board (its move, in the base)."""
        self.misses.append(np.asarray(move, float))
        if self.fits:
            self._facings()

    def _M(self, move):
        return move if self.on_ee else inv_se3(move)

    @staticmethod
    def _unpack(x):
        import cv2
        return rt_to_se3(cv2.Rodrigues(x[:3])[0], x[3:6]), rt_to_se3(cv2.Rodrigues(x[6:9])[0], x[9:12])

    def _project(self, T, points):
        P = points @ T[:3, :3].T + T[:3, 3]
        K = self.K
        return P, P[..., :2] / np.maximum(P[..., 2:3], 1e-6) * (K[0, 0], K[1, 1]) + (K[0, 2], K[1, 2])

    def _residuals(self, x, idx):   # idx: the marker centres fitted (all frames' together, in frame order)
        X, B = self._unpack(x)
        T = inv_se3(X) @ self.Ms @ B                                          # the board in the view, per frame
        k = self.owner[idx]
        P = np.einsum('nij,nj->ni', T[k, :3, :3], self.all_obj[idx]) + T[k, :3, 3]
        return (_project(P, self.K, (0.0, 0.0)) - self.all_img[idx]).ravel()

    def _start(self):
        """A hand-eye start: each frame's board pose by PnP on its marker centres, then AX=XB."""
        import cv2
        boards = []
        for o, i in zip(self.obj, self.img):
            _, rvec, tvec = cv2.solvePnP(o, i, self.K, None)
            boards.append(rt_to_se3(cv2.Rodrigues(rvec)[0], tvec))
        pairs = [(i, j) for i in range(len(boards)) for j in range(i + 1, len(boards))]
        m = self.moves
        if self.on_ee:
            X = solve_axxb([m[j] @ inv_se3(m[i]) for i, j in pairs], [boards[j] @ inv_se3(boards[i]) for i, j in pairs])
            B = _mean_pose([inv_se3(mk) @ X @ b for mk, b in zip(m, boards)])
        else:
            X = solve_axxb([inv_se3(m[j]) @ m[i] for i, j in pairs], [boards[j] @ inv_se3(boards[i]) for i, j in pairs])
            B = _mean_pose([mk @ X @ b for mk, b in zip(m, boards)])
        return np.r_[cv2.Rodrigues(X[:3, :3])[0].ravel(), X[:3, 3], cv2.Rodrigues(B[:3, :3])[0].ravel(), B[:3, 3]]

    def add(self, move, objp, imgp):
        """Add a frame that showed the board (its move, in the base; its markers' corners in the board frame and in view
        pixels, as _annotate returns them), and refit once there are PLAN_MIN_FRAMES."""
        from scipy.optimize import least_squares
        self.moves.append(np.asarray(move, float))
        self.obj.append(objp.reshape(-1, 4, 3).mean(1))
        self.img.append(imgp.reshape(-1, 4, 2).mean(1))
        n = len(self.moves)
        self.owner = np.r_[self.owner, np.full(len(self.obj[-1]), n - 1)]
        if n < PLAN_MIN_FRAMES:
            return
        self.Ms = np.array([self._M(m) for m in self.moves])
        self.all_obj, self.all_img = np.concatenate(self.obj), np.concatenate(self.img)

        def fit(x0, frames):
            return least_squares(self._residuals, x0, args=(np.flatnonzero(np.isin(self.owner, frames)),),
                                 x_scale='jac', loss='soft_l1', f_scale=ROBUST_SCALE_PX).x
        full = fit(self.fits[0] if self.fits else self._start(), list(range(n)))
        folds = min(PLAN_FOLDS, n)
        self.fits = [full] + [fit(full, [k for k in range(n) if k % folds != f]) for f in range(folds)]
        X, B = self._unpack(full)                                             # where the frames so far put the board
        c = (inv_se3(X) @ self.Ms @ B @ self.centre)[:, :3]
        self.landed = c[:, :2] / c[:, 2:3] * (self.K[0, 0], self.K[1, 1]) + (self.K[0, 2], self.K[1, 2])
        self.turns = np.array([m[:3, :3] for m in self.moves])
        self._facings()

    def sees(self, move):
        """Where the board's centre lands in the view (px) if every fit shows the board well with the arm at ``move``
        (the camera link's pose, or the gripper's, in the base): >= min_markers whole markers inside the view,
        PLAN_MARGIN_PX in from its edges and each >= PLAN_MIN_MARKER_PX across, the board facing the view within
        PLAN_MAX_TILT_DEG, and its distance and facing near a frame that showed it and nearer that than one that missed
        it (PLAN_NEAR_M, PLAN_NEAR_DEG). None if any of that fails."""
        M = self._M(move)
        for x in self.fits:
            X, B = self._unpack(x)
            T = inv_se3(X) @ M @ B
            inside, uv = self._inside(T)
            big = np.linalg.norm(uv - np.roll(uv, 1, axis=1), axis=-1).mean(-1) >= PLAN_MIN_MARKER_PX
            if self._facing(T)[2] > PLAN_MAX_TILT_DEG or int((inside & big).sum()) < self.min_markers:
                return None
        X, B = self._unpack(self.fits[0])
        T = inv_se3(X) @ M @ B
        d, n, _ = self._facing(T)
        near = self._nearest(d, n, self.seen_d, self.seen_n)
        if near > 1.0 or near >= self._nearest(d, n, self.missed_d, self.missed_n):
            return None
        return self._project(T, self.centre[None, :3])[1][0]

    def novelty(self, move, landed):
        """How unlike every frame so far a frame at ``move`` whose board lands at ``landed`` (px) would be: to the
        nearest frame, in how the board is turned and where it lands in the view, PLAN_NOVELTY apart counting alike."""
        cos = (np.einsum('nij,ij->n', self.turns, move[:3, :3]) - 1.0) / 2.0              # the turn between them
        turn = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
        return float(np.min(np.hypot(turn / PLAN_NOVELTY[0],
                                     np.linalg.norm(self.landed - landed, axis=1) / PLAN_NOVELTY[1])))

    def summary(self):
        """What the model says (the static view's position, or the fixed board's centre, in the base) and how far its
        refits spread it."""
        if self.on_ee:
            what, c = 'the view', np.array([self._unpack(x)[0][:3, 3] for x in self.fits])
        else:
            what, c = "the board's centre", np.array([(self._unpack(x)[1] @ self.centre)[:3] for x in self.fits])
        (x, y, z), spread = c[0], 1000.0 * float(np.linalg.norm(c.std(0)))
        return (f'{what} at ({x:.3f}, {y:.3f}, {z:.3f}) m in the base, refits within {spread:.0f} mm; the board read '
                f'at {self.seen_d.min():.2f}-{self.seen_d.max():.2f} m, turned {self.seen_tilt.min():.0f}-'
                f'{self.seen_tilt.max():.0f} deg from facing the view, {len(self.missed_d)} misses in view')


def _aimed(planner, draw, move_of):
    """Of PLAN_CANDIDATES waypoints from ``draw()``, one the board model says will show the board well
    (_BoardPlanner.sees; ``move_of(waypoint)``: its move): the one least like the frames so far
    (_BoardPlanner.novelty). None if no candidate is predicted to."""
    best, novelty = None, -1.0
    for _ in range(PLAN_CANDIDATES):
        t = draw()
        move = move_of(t)
        landed = planner.sees(move)
        if landed is None:
            continue
        d = planner.novelty(move, landed)
        if d > novelty:
            best, novelty = t, d
    return best


class _LiveSolve:
    """The calibration, redone as a view jog goes: each camera's view solved (_fit_view) at every frame that shows its
    board, from MIN_VIEWS such frames on, and its errors printed. done() once each view's uncertainty has been
    <= ``target`` px at JOG_STOP_FRAMES of its board frames in a row."""

    def __init__(self, config, models, target):
        self.target = target
        self.cams = {c: (config['cameras'][c]['mount'] == 'gripper', view)
                     for c, (_, _, view) in models.items() if view is not None}
        self.frames = {c: ([], [], []) for c in self.cams}   # its board frames: object points, image points, moves
        self.streak = dict.fromkeys(self.cams, 0)

    def add(self, marks, poses):
        """Add a captured frame (``marks``: _live_annotator's; ``poses``: _save_frame's), and solve and print each view
        it shows the board to."""
        for c, (on_ee, view) in self.cams.items():
            _, objp, imgp = marks.get(c, (0, None, None))
            if objp is None:
                continue
            objs, imgs, moves = self.frames[c]
            objs.append(objp)
            imgs.append(imgp)
            moves.append(poses['ee_in_base' if on_ee else f'parent__{c}'])
            if len(objs) < MIN_VIEWS:
                continue
            try:
                fit = _fit_view(objs, imgs, moves, on_ee, view)
            except Exception as error:                        # a poor early solve must not end the jog
                print(f'    {c}: {len(objs)} board frames, no solve yet ({error})', flush=True)
                continue
            u = fit['uncertainty']
            self.streak[c] = self.streak[c] + 1 if u <= self.target else 0
            fitted = f' ({len(fit["keep"])} fitted)' if len(fit['keep']) < len(objs) else ''
            state = (f'<= {self.target:g} at {self.streak[c]}/{JOG_STOP_FRAMES} board frames in a row'
                     if self.streak[c] else f'target {self.target:g}')
            print(f'    {c}: {len(objs)} board frames{fitted}: rms {fit["rms"]:.2f} px, the arm off by '
                  f'{fit["arm"][0]:.2f} deg / {fit["arm"][1]:.1f} mm per frame; uncertainty {u:.1f} px ({state})', flush=True)

    def unseen(self):
        return [c for c in self.cams if not self.frames[c][0]]

    def done(self):
        return all(s >= JOG_STOP_FRAMES for s in self.streak.values())


def _settle_gripper(ros, joint, target, stop, tolerance=0.03, timeout=5.0):
    """Wait until the gripper ``joint`` measures within ``tolerance`` (rad) of ``target`` and the arm has settled, or
    ``timeout`` s (then a warning; the jog goes on). A stop request raises Stopped."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if stop.is_set():
            raise Stopped('stop requested')
        try:
            if abs(ros.joints(JOINT_MAX_AGE_S)[joint] - target) <= tolerance:
                time.sleep(JOG_SETTLE_S)
                return
        except NotReady:
            pass
        time.sleep(0.05)
    print(f'JOG: the gripper did not report open within {timeout:.0f} s; going on', flush=True)


def _direction_text(way):
    """A unit direction in the arm base (x forward, y the robot's left, z up) in words: "down and to the robot's left"."""
    words = [(2, 'up', 'down'), (0, 'forward', 'back'), (1, "to the robot's left", "to the robot's right")]
    return ' and '.join(pos if way[k] > 0 else neg for k, pos, neg in words if abs(way[k]) > 0.2)


def _jog_capture(config, cams, ros, model, data, bodies, ref, ee, parents, work, amp, present_waist=False,
                 annotate=None, models=None, live=None, start_deg=None):
    """Move the arm to its initial pose (the first camera's ``jog.start_deg``, else ``start_deg``: the serve start
    pose), then drive it through random waypoints around there, saving a frame of every camera at each once the arm has
    settled, until ``live`` (_LiveSolve: each view's solve, redone at every frame that shows its board) is done; then
    back to the initial pose. Without ``live`` it goes on until stopped. With a board on the gripper (a camera with
    ``mount`` "gripper") the gripper opens first and stays open. For a camera on the arm (fixed board) the arm's first
    steps tilt the camera's view up until it shows the board, and the waypoints are around that pose; once the board
    has been seen (with ``annotate`` and ``models``), they aim at poses a model of where the board is predicts will see
    it (_BoardPlanner). Every pose the arm is sent to, and the move there, is checked first (_clearance: the table, and
    the fingertips and the gripper board away from the arm's own links); a waypoint that fails is skipped, and
    JOG_MAX_SKIPS in a row stop the jog. It refuses to start while a camera sends nothing, the joints are stale, or the
    arm already stands where it would not go. Ctrl-C or SIGTERM, the NUC not answering, or stale joints stop it at the
    next setpoint, and the arm holds where it is. Returns how many frames were saved."""
    cj = list(ros.names)
    n_arm = len(cj) - 1                                   # last entry (the gripper) is held, never perturbed
    if len(JOG_HALF_RANGE_DEG) != n_arm:
        raise SystemExit(f'JOG_HALF_RANGE_DEG gives {len(JOG_HALF_RANGE_DEG)} joints; the arm has {n_arm}')
    try:
        joints = ros.joints(JOINT_MAX_AGE_S, wait=6.0)
    except NotReady as error:
        raise SystemExit(str(error)) from None
    missing = [j for j in cj if j not in joints]
    if missing:
        raise SystemExit(f'command joints missing from the robot state: {missing}')
    try:                                    # a silent camera would make every waypoint useless: check before anything
        ros.grab(cams, wait=5.0)            # moves
    except NotReady as error:
        raise SystemExit(f'{error}: every camera must send frames before the arm moves') from None
    start = np.array([joints[j] for j in cj], float)          # where the jog begins and ends
    home = start.copy()                                      # what it samples around (where the board is seen, below)
    lims = _arm_limits(model, cj)
    low_bodies = sorted(({ee} | set(parents.values())) - {ref})   # protect tcp + camera links from the table
    _BOARD_CORNERS.clear()                                   # the gripper board joins the checks once it is modelled

    def blocked(a, b):          # what makes the move from a to b unsafe somewhere along it, or None
        return _path_blocked(model, data, cj, a, b, bodies, ref, ee, low_bodies, JOG_FLOOR_Z)

    poses_now = fk_poses(model, data, joints, bodies)
    low_now = _clearance(poses_now, ref, ee, low_bodies, JOG_FLOOR_Z)   # where the arm stands is not a safe pose
    starts = [(c, (config['cameras'][c].get('jog') or {}).get('start_deg')) for c in cams]
    starts = [(c, deg) for c, deg in starts if deg is not None]
    if not starts and start_deg is not None:             # every calibration starts from the same pose
        starts = [('the initial pose', start_deg)]
    if low_now and not starts:
        raise SystemExit(f'where the arm stands, {low_now}: start the jog from a higher, more open pose')
    if starts:                      # a camera's configured start: the arm goes there first
        first_cam, deg = starts[0]
        if len(deg) != n_arm:
            raise SystemExit(f'{first_cam}: "jog": {{"start_deg"}} needs the {n_arm} arm joints (deg)')
        home[:n_arm] = np.radians(deg)
        near = [cj[k] for k in range(n_arm)
                if not lims[k][0] + JOG_MARGIN_RAD <= home[k] <= lims[k][1] - JOG_MARGIN_RAD]
        if near:
            raise SystemExit(f'{first_cam}: its jog start puts {near} within {math.degrees(JOG_MARGIN_RAD):.0f} deg of '
                             'their limits')
        if low_now:                 # recover from a low pose: the move there may not go lower than the arm is now
            why = _clearance(fk_poses(model, data, {cj[k]: float(home[k]) for k in range(len(cj))}, bodies),
                             ref, ee, low_bodies, JOG_FLOOR_Z)
            if why is None:
                w2b = inv_se3(poses_now[ref])
                lowest = min(float((w2b @ poses_now[b])[2, 3]) for b in low_bodies)
                floor = None if JOG_FLOOR_Z is None else min(JOG_FLOOR_Z, lowest - 0.01)
                why = _path_blocked(model, data, cj, start, home, bodies, ref, ee, low_bodies, floor)
            if why:
                raise SystemExit(f'where the arm stands, {low_now}; the way up to {first_cam} is not clear either '
                                 f'({why}): move the arm higher by hand (Desk) and run again')
            print(f'JOG: where the arm stands, {low_now}; the move to {first_cam} goes no lower than that',
                  flush=True)
        else:
            why = blocked(start, home)
            if why:
                raise SystemExit(f'{first_cam}: on the move to its jog start, {why}: start from another pose')
        print(f'JOG: {first_cam}: the arm goes to its jog start first ({[round(float(v), 2) for v in deg]} deg)',
              flush=True)
    # present-to-waist: on ~JOG_BIAS_FRAC of waypoints, RETRACT the gripper toward a camera by the base so its board
    # fills more of the frame. A camera on the arm (fixed board): every waypoint keeps it turned at most
    # JOG_MAX_TURN_DEG from where it looks once it sees its board.
    home_jd = {**joints, **{cj[k]: float(home[k]) for k in range(len(cj))}}
    fixed_cam = next((c for c in cams if config['cameras'][c].get('mount') == 'fixed'), None)
    hold_open = any(config['cameras'][c].get('mount') == 'gripper' for c in cams)   # a board rides on the gripper
    pj, pj_sign, bias_deg, base_min, turn, search = None, 1.0, None, None, None, None
    if present_waist:
        pj, pj_sign = _dominant_joint(model, data, home_jd, cj, n_arm,
                                      _tcp_range_fn(model, data, bodies, ref, ee))
        bias_deg, base_min = JOG_PRESENT_RETRACT_DEG, JOG_BASE_MIN
        print(f'JOG: present-to-waist -- retract via {cj[pj]} (sign {pj_sign:+.0f}) on '
              f'~{int(JOG_BIAS_FRAC * 100)}% of waypoints (tcp kept >= {JOG_BASE_MIN} m from base)',
              flush=True)
    elif fixed_cam is not None:     # a camera on the arm: tilt its view up until it sees its board (below)
        parent = parents[fixed_cam]
        search = (fixed_cam, parent, _view_axes(fixed_cam, config['cameras'][fixed_cam]))
        print(f'JOG: {fixed_cam}: first tilting its view up {JOG_SEARCH_STEP_DEG:.0f} deg at a time, the shoulder '
              'first, until it sees its board', flush=True)
        turn = _turn_limit(model, data, cj, home, bodies, ref, parent, JOG_MAX_TURN_DEG)
    half = np.radians(JOG_HALF_RANGE_DEG) * float(amp)

    def around(h):              # each arm joint's range for the waypoints around home h
        return (np.array([max(lims[k][0] + JOG_MARGIN_RAD, h[k] - half[k]) for k in range(n_arm)]),
                np.array([min(lims[k][1] - JOG_MARGIN_RAD, h[k] + half[k]) for k in range(n_arm)]))
    lo, hi = around(home)
    planners = {}                         # each view's board model: the waypoints aim at the board once it has one
    if annotate is not None and models:
        planners = {c: _BoardPlanner(models[c][2], models[c][0], config['cameras'][c].get('mount') == 'gripper',
                                     (config['cameras'][c].get('jog') or {}).get('aim_min_markers', PLAN_MIN_MARKERS))
                    for c in cams if models.get(c, (None,) * 3)[2] is not None}
    aim_turn = None if turn is None else (turn[0], turn[1], math.radians(PLAN_MAX_TURN_DEG))
    rng = np.random.default_rng(0)

    def draw(frac, limit):      # a random safe waypoint, within frac of each joint's range around home
        t = last.copy()
        t[:n_arm] = home[:n_arm] + (rng.uniform(lo, hi) - home[:n_arm]) * frac
        if pj is not None and rng.random() < JOG_BIAS_FRAC:
            off = pj_sign * math.radians(rng.uniform(*bias_deg))
            t[pj] = float(np.clip(home[pj] + off, lims[pj][0] + JOG_MARGIN_RAD, lims[pj][1] - JOG_MARGIN_RAD))
        return _make_safe(model, data, cj, t, home, bodies, ref, ee, low_bodies, JOG_FLOOR_Z, base_min, limit)

    def move_of(t, c):          # what carries camera c's board with the arm at t: the camera's link, or the gripper
        poses = fk_poses(model, data, {cj[k]: float(t[k]) for k in range(len(cj))}, bodies)
        return inv_se3(poses[ref]) @ poses[ee if planners[c].on_ee else parents[c]]

    def visit(target, how):     # go there (the move checked first), settle, save a frame: its markers per camera
        nonlocal last, saved, i, skips                       # (None: skipped)
        i += 1
        why = blocked(last, target)
        if why:
            skips += 1
            print(f'  waypoint {i}: skipped, the arm not moved: on the way there {why}', flush=True)
            if skips >= JOG_MAX_SKIPS:
                raise Stopped(f'{skips} waypoints in a row were not clear of the table and the arm')
            return None
        skips = 0
        last = _move(ros, last, target, stop)
        arrived = time.monotonic()
        time.sleep(JOG_SETTLE_S)
        try:                                      # frames that arrived after the settle show the settled arm
            marks, poses = _save_frame(ros, model, data, cams, ref, ee, parents, bodies, work, saved,
                                       since=arrived + JOG_SETTLE_S, wait=FRAME_WAIT_S, annotate=annotate)
        except NotReady as error:
            print(f'  waypoint {i}: skipped ({error})', flush=True)
            return None
        saved += 1
        print(f'  waypoint {i}: captured (saved {saved}){_markers_line(marks)}{how}', flush=True)
        for c, pl in planners.items():
            _, objp, imgp = marks.get(c, (0, None, None))
            move = poses['ee_in_base' if pl.on_ee else f'parent__{c}']
            if objp is None:
                pl.miss(move)
                continue
            had = bool(pl.fits)
            pl.add(move, objp, imgp)
            if pl.fits and pl.on_ee:              # the board's corners join the clearance checks
                _note_board(pl)
            if pl.fits and (not had or len(pl.moves) % 10 == 0):
                print(f'JOG: {c}: board model from {len(pl.moves)} frames: {pl.summary()}; aiming the '
                      'waypoints at it', flush=True)
        if live is not None:
            live.add(marks, poses)
            if saved == 10:
                for c in live.unseen():
                    print(f'JOG: no frame has shown {c} its board yet: is the board where the view looks? '
                          f'(annot/{c}/ has each view)', flush=True)
        return marks

    def go_back():              # to the start, the way checked: straight there, else through home; else the arm holds
        nonlocal last
        for path in ([start], [home, start]):
            hops = [last, *path]
            if all(blocked(a, b) is None for a, b in zip(hops, hops[1:])):
                for q in path:
                    last = _move(ros, last, q, stop)
                return
        print('JOG: neither the move straight back to the start nor the one through home is clear of the table and the '
              'arm: the arm holds here; bring it back with the DROID GUI or by hand (Desk)', flush=True)
    stop = threading.Event()
    handlers = {s: signal.signal(s, lambda *_: stop.set()) for s in (signal.SIGINT, signal.SIGTERM)}
    requester = 'the calibration jog'
    saved, i, last, skips = 0, 0, start.copy(), 0
    try:
        ros.claim(requester)
        try:
            if hold_open:           # the board rides on the gripper: open it; every setpoint's last entry keeps it so
                opened = last.copy()
                opened[-1] = float(config['motion']['gripper_joint_range'][0])
                print('JOG: the gripper opens and stays open (the board rides on it)', flush=True)
                last = _move(ros, last, opened, stop)
                _settle_gripper(ros, cj[-1], opened[-1], stop)
                start[-1] = home[-1] = opened[-1]
            if not np.allclose(home, start):
                last = _move(ros, last, home, stop)
                start = home.copy()                      # the jog comes back to its initial pose, not where it stood
            if hold_open and annotate is not None:     # bring the board into the views: along jog.search_dir until seen
                board_cams = [c for c in cams if config['cameras'][c].get('mount') == 'gripper']
                names = ', '.join(board_cams)
                seen = lambda m: m is not None and all(m.get(c, (0, None, None))[1] is not None for c in board_cams)
                search_jog = config['cameras'][board_cams[0]].get('jog') or {}
                raw_way = search_jog.get('search_dir', (1.0, 0.0, 1.0))
                reach_max = float(search_jog.get('search_max_m', JOG_REACH_MAX_M))   # e.g. shorter toward a camera
                if not reach_max > 0:
                    raise SystemExit(f'{board_cams[0]}: "jog": {{"search_max_m"}} must be a positive distance (m)')
                way = np.asarray(raw_way, float)                      # in the base: x forward, y left, z up
                if way.shape != (3,) or not np.linalg.norm(way) > 0:
                    raise SystemExit(f'{board_cams[0]}: "jog": {{"search_dir"}} needs a nonzero [x, y, z] in the base')
                way /= np.linalg.norm(way)
                dir_text = _direction_text(way)
                poses = fk_poses(model, data, {cj[k]: float(last[k]) for k in range(len(cj))}, bodies)
                first = inv_se3(poses[ref]) @ poses[ee]                # the tcp at the start, in the base
                print(f'JOG: {names}: first moving the gripper {dir_text} {100 * JOG_REACH_STEP_M:.0f} cm at a '
                      'time, its orientation held, until the view sees the board', flush=True)
                marks = visit(last.copy(), ' (as it starts, looking for the board)')
                found, why, k = (last.copy() if seen(marks) else None), '', 0
                while found is None:
                    k += 1
                    if k * JOG_REACH_STEP_M > reach_max + 1e-9:
                        why = f'{100 * reach_max:.0f} cm {dir_text} already'
                        break
                    goal = first.copy()
                    goal[:3, 3] = first[:3, 3] + way * (k * JOG_REACH_STEP_M)
                    jd = {**joints, **{cj[j]: float(last[j]) for j in range(len(cj))}}
                    t = last.copy()
                    t[:n_arm], off_m, off_deg = _reach(model, data, cj, n_arm, jd, bodies, ref, ee, goal, low_bodies,
                                                       lims)
                    if off_m > 0.002 or off_deg > 1.0:
                        why = 'the arm reaches no farther that way'
                        break
                    blocker = blocked(last, t)
                    if blocker:
                        why = f'on the next step {blocker}'
                        break
                    marks = visit(t, f' (the gripper {100 * k * JOG_REACH_STEP_M:.0f} cm {dir_text}, looking for '
                                     'the board)')
                    if seen(marks):
                        found = last.copy()
                if found is None:
                    print(f'JOG: {names}: the view never saw the board ({why}): start with the board where the camera '
                          'sees it (annot/ has each view) and run again; returning to the start', flush=True)
                    go_back()
                    return saved
                home = found
                lo, hi = around(home)
                where = (f'the gripper {100 * k * JOG_REACH_STEP_M:.0f} cm {dir_text}' if k
                         else 'where the jog starts')
                print(f'JOG: {names}: the view sees the board, {where}; the waypoints are around there', flush=True)
            if search is not None:  # tilt the view up (toward its image top) until it sees its board
                cam, parent, axes = search
                step = math.radians(JOG_SEARCH_STEP_DEG)
                view_at = lambda q: _turn_limit(model, data, cj, q, bodies, ref, parent, 0.0)[1] @ axes
                first = view_at(last)

                def tilted(q):      # how far the view at q is turned up from where it started, deg
                    f = view_at(q)[:, 2]
                    return math.degrees(math.atan2(-(first[:, 1] @ f), first[:, 2] @ f))

                def tilt_ups(q):    # ways to turn the view up a step from q, in order: (target, what turns it)
                    now = view_at(q)
                    out = []
                    for k, name in JOG_SEARCH_JOINTS:       # one joint alone: exactly a step, whichever way is up
                        for sign in (1.0, -1.0):
                            t = q.copy()
                            t[k] += sign * step
                            if (view_at(t)[:, 2] @ -now[:, 1] > math.sin(step) / 2       # turns it up, at least half
                                    and lims[k][0] + JOG_MARGIN_RAD <= t[k] <= lims[k][1] - JOG_MARGIN_RAD):
                                out.append((t, name))
                                break
                    jd = {**joints, **{cj[k]: float(q[k]) for k in range(len(cj))}}
                    t = q.copy()
                    t[:n_arm], off, _, _ = _point_view(model, data, cj, n_arm, jd, bodies, ref, ee, parent, axes[:, 2],
                                                       now @ (0.0, -math.sin(step), math.cos(step)), low_bodies, lims)
                    if off <= JOG_SEARCH_TOL_DEG:
                        out.append((t, 'the gripper held'))
                    return out

                found, why = None, ''
                marks = visit(last.copy(), f' (the {cam} view as it starts, looking for its board)')
                if marks and marks.get(cam, (0, None, None))[1] is not None:
                    found = last.copy()
                while found is None:
                    moved, why = False, ''
                    for t, name in tilt_ups(last):
                        if -view_at(t)[2, 2] < math.sin(math.radians(JOG_SEARCH_MIN_BELOW_DEG)):
                            why = (f'tilted further, it would look within {JOG_SEARCH_MIN_BELOW_DEG:.0f} deg of the '
                                   'horizon')
                            break
                        blocker = blocked(last, t)
                        if blocker:
                            why = f'every way to tilt it further is blocked: {blocker}'
                            continue
                        marks = visit(t, f' (the {cam} view tilted up {tilted(t):.0f} deg by {name}, looking for its '
                                         'board)')
                        moved = True
                        if marks and marks.get(cam, (0, None, None))[1] is not None:
                            found = last.copy()
                        break
                    if not moved:
                        why = why or 'the arm can tilt it no further (its joint limits)'
                        break
                tilt = tilted(last)
                if found is None:
                    print(f'JOG: {cam}: its view never saw its board ({why}): put the board where it can see it '
                          '(annot/ has each view) and run again; returning to the start', flush=True)
                    go_back()
                    return saved
                home = found
                lo, hi = around(home)
                turn = _turn_limit(model, data, cj, home, bodies, ref, parent, JOG_MAX_TURN_DEG)
                aim_turn = (turn[0], turn[1], math.radians(PLAN_MAX_TURN_DEG))
                print(f'JOG: {cam}: its view sees its board, tilted up {tilt:.0f} deg; waypoints keep the camera '
                      f'within {JOG_MAX_TURN_DEG:.0f} deg of that', flush=True)
            until = (f'each view\'s uncertainty has been <= {live.target:g} px at {JOG_STOP_FRAMES} of its board '
                     'frames in a row (Ctrl-C ends it sooner)' if live is not None else 'Ctrl-C')
            print(f'JOG: waypoints until {until}, around {np.round(np.degrees(home[:n_arm]), 1).tolist()} deg '
                  f'(slew {math.degrees(JOG_SLEW_RAD_S):.0f} deg/s, amp {amp})', flush=True)
            while True:
                target, how = None, ''
                ready = [c for c, pl in planners.items() if pl.fits and not (live and live.streak[c] >= JOG_STOP_FRAMES)]
                if i == 0:                                # first, where the board is in view: home
                    target, how = last.copy(), ' (home)'
                elif ready:                               # aim for the view with the fewest board frames
                    c = min(ready, key=lambda c: len(planners[c].moves))
                    target = _aimed(planners[c], lambda: draw(1.0, aim_turn), lambda t: move_of(t, c))
                    how = ((f' (aimed at {c}\'s board)' if len(planners) > 1 else ' (aimed at the board)')
                           if target is not None else ' (exploring: no pose predicted to show the board)')
                if target is None:
                    target = draw(JOG_EXPLORE_FRAC, turn)
                marks = visit(target, how)
                if marks is None:
                    continue
                if live is not None and live.done():
                    print(f'JOG: each view\'s uncertainty has been <= {live.target:g} px at '
                          f'{JOG_STOP_FRAMES} board frames in a row: done', flush=True)
                    break
            print('JOG: returning to the start', flush=True)
            go_back()
        except Stopped as error:
            _hold(ros, cj)
            print(f'JOG STOPPED: {error}; the arm holds where it is', flush=True)
        finally:
            ros.release(requester)
    finally:
        for s, handler in handlers.items():
            signal.signal(s, handler)
    return saved


def capture(config, settings, cams, work, interval, timeout, target, jog=False, amp=1.0, present_waist=False,
            others=None, robot=None):
    """Open the robot (droid.Droid: the NUC's control server and the cameras ``cams``; or ``robot``, a stand-in),
    complete the cameras' config entries as the server does (droid.complete: the raw lens from the ZED SDK, the view's K
    from its fov_deg, a calibration as the prior), and record frames of ``cams`` into ``work``, each with the joints and
    the FK poses the solve needs: with ``jog`` at waypoints (drives the arm, _jog_capture), else every ``interval`` s
    while the board is moved by hand. Each frame's annotated view is written as it is captured (_live_annotator) and
    each view solved as it goes (_LiveSolve); the capture ends when every view's uncertainty has come down to ``target``
    px, on Ctrl-C or SIGTERM, or (without ``jog``) after ``timeout`` s without a new frame. ``settings``: the serve
    config, whose node graph gives the TCP. Returns (the completed entries, the capture's meta), both also written into
    ``work`` (cameras.json, meta.json) for --reuse-capture."""
    import mujoco
    global JOG_FLOOR_Z
    ref, ee = config['reference_body'], config['ee_body']
    parents = {c: config['cameras'][c]['parent_body'] for c in cams}
    keepout = {b for first, second, _, _ in JOG_KEEPOUT for b in (first, second)}
    bodies = sorted({ref, ee, *parents.values(), *keepout})
    JOG_FLOOR_Z = None if config.get('table_z_m') is None else float(config['table_z_m']) + JOG_TABLE_CLEARANCE_M
    tcp = next(n for n in settings['node_graph']['nodes'] if n.get('tcp'))
    if robot is None:
        from droid_deploy.droid import Droid
        robot = Droid(config, command=jog, cams=cams)
    ros = robot
    try:
        entries = {c: json.loads(json.dumps(ros.config['cameras'][c])) for c in cams}
        models = {c: (entries[c]['board'], *_models(c, entries[c], need_view=True)) for c in cams}
        urdf = ros.wait_urdf()
        model = build_model(urdf, tcp=(tcp['body'], tcp.get('site_position', (0.0, 0.0, 0.0))))
        data = mujoco.MjData(model)
        work = Path(work); work.mkdir(parents=True, exist_ok=True)
        meta = {'camera_names': cams, 'parent_body': parents, 'reference_body': ref, 'ee_body': ee,
                'captured': time.strftime('%Y-%m-%d %H:%M'), 'table_z_m': config.get('table_z_m')}
        (work / 'robot.urdf').write_text(urdf)
        (work / 'meta.json').write_text(json.dumps(meta, indent=2))
        (work / 'cameras.json').write_text(json.dumps(entries, indent=2))
        annotate = _live_annotator(work, models, others or {})
        live = _LiveSolve(config, models, target)
        live = live if live.cams else None
        if jog:
            completed = {**config, 'cameras': {**config['cameras'], **entries}}     # with the priors (wrist search)
            saved = _jog_capture(completed, cams, ros, model, data, bodies, ref, ee, parents, work, amp,
                                 present_waist=present_waist, annotate=annotate, models=models, live=live,
                                 start_deg=settings.get('start_pose_deg'))
        else:
            stop = threading.Event()
            handlers = {s: signal.signal(s, lambda *_: stop.set()) for s in (signal.SIGINT, signal.SIGTERM)}
            try:
                saved, last = 0, 0.0
                deadline = time.monotonic() + timeout
                while not stop.is_set() and time.monotonic() < deadline:
                    try:
                        marks, poses = _save_frame(ros, model, data, cams, ref, ee, parents, bodies, work, saved,
                                                   since=last, annotate=annotate)
                    except NotReady:
                        time.sleep(0.1)
                        continue
                    saved, last = saved + 1, time.monotonic()
                    print(f'  frame {saved}{_markers_line(marks)}', flush=True)
                    if live is not None:
                        live.add(marks, poses)
                        if live.done():
                            print(f'each view\'s uncertainty has been <= {live.target:g} px at '
                                  f'{JOG_STOP_FRAMES} board frames in a row: done', flush=True)
                            break
                    deadline = last + timeout
                    time.sleep(interval)
                if stop.is_set():
                    print('capture stopped (Ctrl-C or SIGTERM)', flush=True)
            finally:
                for s, handler in handlers.items():
                    signal.signal(s, handler)
        if saved == 0:
            raise SystemExit('no frames captured')
        print(f'captured {saved} frames -> {work}', flush=True)
        for c in cams:
            _write_mp4(c, work)
    finally:
        ros.close()
    return entries, meta


# ---- hand-eye AX=XB ---------------------------------------------------------------
def _log_so3(R):
    import cv2
    return cv2.Rodrigues(R)[0].reshape(3)


def solve_axxb(A_list, B_list):
    M = np.zeros((3, 3))
    for A, B in zip(A_list, B_list):
        M += np.outer(_log_so3(A[:3, :3]), _log_so3(B[:3, :3]))
    U, _, Vt = np.linalg.svd(M)
    Rx = U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    C, d = [], []
    for A, B in zip(A_list, B_list):
        C.append(A[:3, :3] - np.eye(3))
        d.append(Rx @ B[:3, 3] - A[:3, 3])
    tx, *_ = np.linalg.lstsq(np.vstack(C), np.concatenate(d), rcond=None)
    return rt_to_se3(Rx, tx)


# ---- solve one camera -------------------------------------------------------------
MIN_MARKERS = 4          # aruco markers a frame needs to be usable (for a view: wholly inside it)
MIN_VIEWS = 6            # usable frames needed to calibrate a camera
BOARD_FIT_PX = 4.0       # a marker belongs to a board only if its corners fit one pose of it, within this (view pixels)
OUTLIER_FACTOR, OUTLIER_ROUNDS = 2.5, 3   # drop frames > 2.5x the median disagreement, refit, <= 3 times
ROBUST_SCALE_PX = 1.0    # the fits' robust loss: a marker corner off by more than this (view pixels) counts less
PREDICTION_DEPTHS_M = {'gripper': (0.3, 0.6, 1.0),   # the uncertainty's test points: a camera in the scene sees the
                       'fixed': (0.1, 0.25, 0.5)}    # workspace from 0.3-1 m, one on the arm (fixed board) from 0.1-0.5 m
UNCERTAINTY_FOLDS = 10   # the uncertainty: refits, each without one of this many shares of the frames
ARM_ERROR_START = (math.radians(0.5), 0.003)   # the arm's pose error per frame (rad, m) the fit starts from and then
ARM_ERROR_MIN = (math.radians(0.005), 5e-5)    # measures (ARM_ROUNDS refits at most), never below ARM_ERROR_MIN
ARM_ROUNDS = 5
MAX_UNCERTAINTY_PX = 1.0 # --uncertainty-px's default: a capture goes on until the refits disagree by at most this on
                         # where points land, and a solve above it warns
HANDEYE_MAX_PAIRS = 2000 # the hand-eye start uses at most this many pairs of frames (a fixed random choice)
REVIEW_SCALE = 2         # the annotated view frames are drawn at this multiple of the view's size


def _board(spec):
    import cv2
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec['dict']))
    square = spec['square_len_mm'] / 1000.0
    marker = square * spec.get('marker_ratio', 0.75)   # marker size derived from the one measured number
    board = cv2.aruco.CharucoBoard((spec['squares_x'], spec['squares_y']), square, marker, d)
    obj_by_id = {int(mid): np.asarray(board.getObjPoints()[k])
                 for k, mid in enumerate(np.asarray(board.getIds()).flatten())}
    return cv2.aruco.ArucoDetector(d), obj_by_id


def _claim(ids, points, usable, layouts, K):
    """Which of a frame's markers belong to which board: both boards number their markers from 0, so only where the
    markers sit tells them apart. Per board (``layouts``: its object points by id), the largest set of the markers with
    its ids whose corners fit one pose of it (RANSAC PnP through the view's K, within BOARD_FIT_PX); boards with larger
    sets claim first, and a board keeps only a set of at least MIN_MARKERS. ``points``: each marker's corners in view
    pixels; ``usable``: whether they may take part. Returns each board's set of marker indices."""
    import cv2
    found = []
    for layout in layouts:
        cand = [k for k, m in enumerate(ids) if m in layout and usable[k]]
        best = set()
        if len(cand) >= MIN_MARKERS:
            cv2.setRNGSeed(0)                                    # a frame always splits the same way
            ok, _, _, inliers = cv2.solvePnPRansac(
                np.concatenate([layout[ids[k]] for k in cand]).astype(np.float64),
                np.concatenate([points[k] for k in cand]).astype(np.float64), K, None,
                iterationsCount=500, reprojectionError=BOARD_FIT_PX, flags=cv2.SOLVEPNP_AP3P)
            if ok and inliers is not None:
                inl = set(inliers.ravel().tolist())
                best = {k for j, k in enumerate(cand) if all(4 * j + c in inl for c in range(4))}
        found.append(best)
    claimed, out = set(), [set() for _ in layouts]
    for b in sorted(range(len(layouts)), key=lambda b: -len(found[b])):
        mine = found[b] - claimed
        if len(mine) >= MIN_MARKERS:
            out[b] = mine
            claimed |= mine
    return out


def _annotate(cam, img, i, board, raw, view=None, maps=None, others=()):
    """This board's markers in one raw frame (``board``: its detector and object points by id), drawn onto the view
    when there is one (``maps`` renders it) or else the raw frame: markers used green, the rest red. With a view, a
    marker counts only if it fits one pose of this board together with the others counted (_claim; markers that fit one
    of the ``others`` boards, their object points by id, go to it), and only markers wholly inside the view are used.
    Without a view (a lens capture) only this board's ids seen once count. Returns (the drawing, BGR; how many markers
    are used; their object points and image points, or None, None with fewer than MIN_MARKERS)."""
    import cv2
    detector, obj_by_id = board
    # Calibrate from aruco MARKER corners (reliable on a 180-deg fisheye), not interpolated ChArUco corners -- the
    # ChArUco planar corner interpolation collapses under heavy fisheye distortion.
    corners, ids, _ = detector.detectMarkers(img)
    flat = [] if ids is None else [int(m) for m in ids.flatten()]
    # keep only THIS board's markers, each seen once: drops false positives (stray ids in the busy scene) and ids that
    # collide with the other board (both boards share DICT_5X5 ids 0..N)
    theirs = set()
    if view is None:
        count = Counter(flat)
        mine = {k for k, m in enumerate(flat) if m in obj_by_id and count[m] == 1}
        points, shown = [c.reshape(4, 2) for c in corners], [True] * len(flat)
        vis, draw_scale = img, 1
    else:
        mapped = [_to_view(c.reshape(4, 2), raw, view, bounds=False) for c in corners]
        points = [uv for uv, _ in mapped]
        (w, h), inside = view['size'], []
        usable = []                                  # in front of the view, within a view's size of it
        for uv, ahead in mapped:
            inside.append(bool(ahead.all() and ((uv >= -0.5) & (uv <= (w - 0.5, h - 0.5))).all()))
            usable.append(bool(ahead.all() and ((uv > (-w, -h)) & (uv < (2 * w, 2 * h))).all()))
        claims = _claim(flat, points, usable, [obj_by_id, *others], view['K'])
        mine, theirs = claims[0], set().union(*claims[1:])
        shown = inside
        vis, draw_scale = cv2.remap(img, *maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT), REVIEW_SCALE
    used = [k for k in range(len(flat)) if k in mine and shown[k]]
    ignored = [k for k in range(len(flat)) if k not in mine and shown[k]]
    vis = cv2.cvtColor(np.asarray(vis), cv2.COLOR_RGB2BGR).copy()
    for keys, colour in ((ignored, (0, 0, 255)), (used, (0, 200, 0))):
        if keys:
            cv2.aruco.drawDetectedMarkers(
                vis, [((points[k] + 0.5) * draw_scale - 0.5).reshape(1, 4, 2).astype(np.float32) for k in keys],
                np.array([[flat[k]] for k in keys]), colour)
    scale = vis.shape[1] / 1280.0
    other = f', {len(theirs)} fit another board' if theirs else ''
    cv2.putText(vis, f'{cam} step{i}: {len(used)} used (green) / {len(flat)} detected  (red = ignored{other})',
                (max(4, int(14 * scale)), max(12, int(40 * scale))), cv2.FONT_HERSHEY_SIMPLEX,
                max(0.3, 0.95 * scale), (0, 255, 255), 2 if scale > 0.6 else 1, cv2.LINE_AA)
    if len(used) < MIN_MARKERS:
        return vis, len(used), None, None
    return (vis, len(used), np.concatenate([obj_by_id[flat[k]] for k in used]).astype(np.float64).reshape(-1, 1, 3),
            np.concatenate([points[k] for k in used]).astype(np.float64).reshape(-1, 1, 2))


def _detect(cam, work, frames, board_spec, raw, view=None, others=()):
    """This board's markers in every frame: (frame indices, object points, image points) of the frames that show at
    least MIN_MARKERS of them. In raw pixels, or with a ``view`` in view pixels, keeping only the markers wholly inside
    the view, and only those that fit this board rather than one of the ``others`` (board specs). Draws every frame
    into work/annot/<cam>/ (_annotate)."""
    import cv2
    board = _board(board_spec)
    others = [_board(spec)[1] for spec in others]
    maps = None if view is None else _view_maps(raw, view, REVIEW_SCALE)
    adir = Path(work) / 'annot' / cam
    adir.mkdir(parents=True, exist_ok=True)
    good, objp, imgp = [], [], []
    for i, fr in enumerate(frames):
        vis, _, op, ip = _annotate(cam, fr[f'raw__{cam}'], i, board, raw, view, maps, others)
        cv2.imwrite(str(adir / f'step_{i:04d}.png'), vis)
        if op is not None:
            good.append(i)
            objp.append(op)
            imgp.append(ip)
    _write_mp4(cam, work, subdir='annot', suffix='_detected')   # debug video even if the solve fails below
    if len(good) < MIN_VIEWS:
        where = '' if view is None else ' inside the view'
        raise SystemExit(f'only {len(good)} frames with >= {MIN_MARKERS} markers{where}; need >= {MIN_VIEWS}')
    return good, objp, imgp


def _live_annotator(work, models, others):
    """A function (idx, frames) -> {camera: (markers used, their object points, their image points; see _annotate)}
    that draws each captured frame into work/annot/<cam>/ as it is saved, as the solve will (_annotate), so a capture
    can be watched while it runs. ``models``: {camera: (board spec, raw lens, view or None)}; ``others``: {camera: the
    other boards' specs}."""
    import cv2
    ctx = {cam: (_board(spec), raw, view, None if view is None else _view_maps(raw, view, REVIEW_SCALE),
                 [_board(o)[1] for o in others.get(cam, ())])
           for cam, (spec, raw, view) in models.items()}
    for cam in ctx:
        (Path(work) / 'annot' / cam).mkdir(parents=True, exist_ok=True)

    def annotate(idx, frames):
        marks = {}
        for cam, (board, raw, view, maps, other) in ctx.items():
            vis, *marks[cam] = _annotate(cam, frames[cam], idx, board, raw, view, maps, other)
            cv2.imwrite(str(Path(work) / 'annot' / cam / f'step_{idx:04d}.png'), vis)
        return marks
    return annotate


def _markers_line(marks):
    """'; markers used: waist 9, ...' for a progress line, or '' without live annotation."""
    return '; markers used: ' + ', '.join(f'{cam} {m[0]}' for cam, m in marks.items()) if marks else ''


def _fit_lens(cam, lens, size, objp, imgp):
    """The raw lens (``lens``: its model) fitted to the markers: (K, D, rms in raw pixels)."""
    import cv2
    w, h = size
    if lens == 'pinhole':                                # a rectilinear lens: OpenCV's standard model, no k3
        rms, K, D, _, _ = cv2.calibrateCamera([o.reshape(-1, 3).astype(np.float32) for o in objp],
                                              [i.reshape(-1, 2).astype(np.float32) for i in imgp], (w, h), None, None,
                                              flags=cv2.CALIB_FIX_K3)
    else:
        f0 = w / math.pi                                   # 180-deg equidistant fisheye: f ~= width/pi
        K = np.array([[f0, 0, w / 2.0], [0, f0, h / 2.0], [0, 0, 1.0]])
        flags = (cv2.fisheye.CALIB_USE_INTRINSIC_GUESS | cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
                 | cv2.fisheye.CALIB_FIX_SKEW)
        rms, K, D, _, _ = cv2.fisheye.calibrate(
            objp, imgp, (w, h), K, np.zeros((4, 1)), flags=flags,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
    D = np.asarray(D).reshape(-1, 1)
    print(f'  {cam}: {lens} lens rms={rms:.3f}px over {len(objp)} views  f=({K[0,0]:.1f},{K[1,1]:.1f})  '
          f'D={np.round(D.ravel(), 4).tolist()}', flush=True)
    return np.asarray(K), D, float(rms)


def _project(P, K, D, jac=False):
    """(N, 3) points in a camera's optical frame -> (N, 2) pixels, through K and the radial k1, k2 of OpenCV's standard
    model (D[0], D[1]), as cv2.projectPoints with D = (k1, k2, 0, 0, 0). With ``jac``, also the derivatives: by
    (fx, fy, cx, cy, k1, k2), (N, 2, 6), and by the point, (N, 2, 3)."""
    Z = np.maximum(P[:, 2], 1e-9)
    x, y = P[:, 0] / Z, P[:, 1] / Z
    r2 = x * x + y * y
    s = 1.0 + D[0] * r2 + D[1] * r2 * r2
    fx, fy = K[0, 0], K[1, 1]
    uv = np.stack([fx * x * s + K[0, 2], fy * y * s + K[1, 2]], axis=1)
    if not jac:
        return uv
    by_k = np.zeros((len(P), 2, 6))
    by_k[:, 0, 0], by_k[:, 0, 2], by_k[:, 0, 4], by_k[:, 0, 5] = x * s, 1.0, fx * x * r2, fx * x * r2 * r2
    by_k[:, 1, 1], by_k[:, 1, 3], by_k[:, 1, 4], by_k[:, 1, 5] = y * s, 1.0, fy * y * r2, fy * y * r2 * r2
    ds = 2.0 * (D[0] + 2.0 * D[1] * r2)                                  # d s / d(x, y) = ds * (x, y)
    by_xy = np.array([[fx * (s + ds * x * x), fx * ds * x * y], [fy * ds * x * y, fy * (s + ds * y * y)]])
    xy_by_p = np.array([[1.0 / Z, 0.0 * Z, -x / Z], [0.0 * Z, 1.0 / Z, -y / Z]])
    return uv, by_k, np.einsum('abn,bcn->nac', by_xy, xy_by_p)


def _exp_so3(w):
    """(n, 3) rotation vectors -> (n, 3, 3) rotation matrices."""
    S = _skew(w)
    th = np.linalg.norm(w, axis=1)
    small = th < 1e-8
    safe = np.where(small, 1.0, th)
    a = np.where(small, 1.0, np.sin(th) / safe)
    b = np.where(small, 0.5, (1.0 - np.cos(th)) / safe ** 2)
    return np.eye(3) + a[:, None, None] * S + b[:, None, None] * (S @ S)


def _skew(v):
    """(n, 3) vectors -> (n, 3, 3) matrices [v]x, with [v]x u = v x u."""
    S = np.zeros((len(v), 3, 3))
    S[:, 0, 1], S[:, 0, 2], S[:, 1, 2] = -v[:, 2], v[:, 1], -v[:, 0]
    return S - S.transpose(0, 2, 1)


def _bundle(obj, img, M, on_ee, k, X, B, xi, sigma, iterations=100):
    """The view's calibration as one robust least-squares fit (Levenberg-Marquardt) over every marker corner of the
    frames: its intrinsics ``k`` (fx, fy, cx, cy, k1, k2), its pose ``X``, the board's pose ``B``, and per frame a
    correction ``xi`` of the arm's pose (a rotation vector and a shift in m). Where the arm puts the board in the view in
    frame f, inv(X) @ M[f] @ B, is corrected in the camera frame (board fixed, camera on the arm) or in the board frame
    (board on the gripper). The corrections are held near zero by the arm's accuracy ``sigma`` (rad, m): each frame's
    image fixes its own pose, and the arm ties the frames together only as tightly as it deserves. Each step solves the
    corrections out first (a Schur complement), so a step costs little however many frames there are. Returns (k, X, B,
    xi, each correction's variance (frames, 6))."""
    counts = [len(o) for o in obj]
    first = np.r_[0, np.cumsum(counts)[:-1]]                             # each frame's first corner
    owner = np.repeat(np.arange(len(obj)), counts)
    O, I = np.concatenate(obj), np.concatenate(img)
    prior = 1.0 / np.repeat(np.square(sigma), 3)                         # 1 / each correction component's variance
    f2 = ROBUST_SCALE_PX ** 2
    eye = np.broadcast_to(np.eye(3), (len(O), 3, 3))

    def intrinsics(k):
        return np.array([[k[0], 0.0, k[2]], [0.0, k[1], k[3]], [0.0, 0.0, 1.0]]), k[4:6]

    def forward(k, X, B, xi):
        TF = inv_se3(X) @ M @ B                                          # the board in the view, where the arm puts it
        Rx = _exp_so3(xi[:, :3])[owner]
        if on_ee:                                                        # Q: the corrected board point, board frame
            Q = np.einsum('nij,nj->ni', Rx, O) + xi[owner, 3:]
            P = np.einsum('nij,nj->ni', TF[owner, :3, :3], Q) + TF[owner, :3, 3]
        else:                                                            # Q: the point where the arm puts it
            Q = np.einsum('nij,nj->ni', TF[owner, :3, :3], O) + TF[owner, :3, 3]
            P = np.einsum('nij,nj->ni', Rx, Q) + xi[owner, 3:]
        r = _project(P, *intrinsics(k)) - I
        cost = f2 * float(np.sum(np.sqrt(1.0 + r * r / f2) - 1.0)) + 0.5 * float(np.sum(xi * xi * prior))
        return cost, r, P, Q, TF[owner, :3, :3], Rx

    cost, r, P, Q, RF, Rx = forward(k, X, B, xi)
    damping, V = 1e-3, None
    for _ in range(iterations):
        _, by_k, by_p = _project(P, *intrinsics(k), jac=True)
        shift = xi[owner, 3:]
        if on_ee:
            d_xi = RF @ np.concatenate([-_skew(Q - shift), eye], axis=2)
            d_X = np.concatenate([_skew(P), -eye], axis=2)
            d_B = RF @ np.concatenate([-_skew(Q), eye], axis=2)
        else:
            d_xi = np.concatenate([-_skew(P - shift), eye], axis=2)
            d_X = Rx @ np.concatenate([_skew(Q), -eye], axis=2)
            d_B = (Rx @ RF) @ np.concatenate([-_skew(O), eye], axis=2)
        Jg = np.concatenate([by_k, by_p @ d_X, by_p @ d_B], axis=2)      # by k, X, B: (corners, 2, 18)
        Jf = by_p @ d_xi                                                 # by the frame's correction: (corners, 2, 6)
        w = 1.0 / np.sqrt(1.0 + r * r / f2)                              # the robust loss as weights
        Jg2 = Jg.reshape(-1, 18)
        U = (Jg2 * w.reshape(-1, 1)).T @ Jg2
        gg = Jg2.T @ (w * r).reshape(-1)
        V = np.add.reduceat(np.einsum('na,nai,naj->nij', w, Jf, Jf), first) + np.diag(prior)
        W = np.add.reduceat(np.einsum('na,nai,naj->nij', w, Jg, Jf), first)
        gf = np.add.reduceat(np.einsum('na,nai,na->ni', w, Jf, r), first) + xi * prior
        while True:
            Vi = np.linalg.inv(V + damping * V * np.eye(6))
            WVi = W @ Vi
            dg = np.linalg.solve(U + damping * np.diag(np.diag(U)) - np.einsum('nij,nkj->ik', WVi, W),
                                 np.einsum('nij,nj->i', WVi, gf) - gg)
            df = -np.einsum('nij,nj->ni', Vi, gf + np.einsum('nji,j->ni', W, dg))
            k_try, xi_try = k + dg[:6], xi + df
            X_try = X @ rt_to_se3(_exp_so3(dg[None, 6:9])[0], dg[9:12])
            B_try = B @ rt_to_se3(_exp_so3(dg[None, 12:15])[0], dg[15:18])
            trial = forward(k_try, X_try, B_try, xi_try)
            if trial[0] < cost or damping > 1e8:
                break
            damping *= 10.0
        if trial[0] >= cost:
            break
        converged = cost - trial[0] < 1e-10 * cost
        k, X, B, xi = k_try, X_try, B_try, xi_try
        cost, r, P, Q, RF, Rx = trial
        damping = max(damping / 10.0, 1e-9)
        if converged:
            break
    return k, X, B, xi, np.diagonal(np.linalg.inv(V), axis1=1, axis2=2)


def _fit_view(objp, imgp, moves, on_ee, view):
    """Calibrate a view from its board frames, and measure how well the frames pin it down. One fit (_bundle) of its
    intrinsics (fx, fy, cx, cy and the radial k1, k2 of OpenCV's standard model), its pose, the board's pose and each
    frame's correction of the arm's pose, from view.K and a hand-eye start (each frame's board pose by PnP, then AX=XB).
    The arm's accuracy, which holds the corrections near zero, is measured by the fit itself: refitted with the
    corrections' spread until it settles (ARM_ROUNDS). So each image fixes its frame's pose, the arm's pose errors can't
    pull the intrinsics (they land in the corrections), and the arm's motion still pins what the images alone can't
    (the focal length against k1, k2). Frames whose markers fit more than OUTLIER_FACTOR x worse than the median
    frame's, or whose correction is more than OUTLIER_FACTOR x the arm's error, are dropped. The uncertainty: the fit redone UNCERTAINTY_FOLDS times, each without one share of the
    frames (a grouped jackknife), and how far the refits disagree on where points across the view land (95th
    percentile); intrinsics and pose that trade off against each other don't count.

    ``objp``, ``imgp``: each frame's marker corners, in the board frame and in view pixels. ``moves``: each frame's pose,
    in the base, of what carries the board through the view: the gripper (``on_ee``: the board on it, the camera static)
    or the camera's parent body (the board fixed). Returns {"K", "D"; "X": the view in the base (on_ee) or in its parent;
    "B": the board in the gripper (on_ee) or in the base; "keep": the frames used; "rms": the fit (view pixels); "rms_arm":
    the same without the corrections; "arm": the arm's error per frame (deg, mm); "uncertainty" (px) over "folds"
    refits; "placed": each kept frame's board position by its own corrected pose, in B's frame}."""
    import cv2
    n = len(objp)
    obj = [np.asarray(o, float).reshape(-1, 3) for o in objp]
    img = [np.asarray(p, float).reshape(-1, 2) for p in imgp]
    moves = [np.asarray(m, float) for m in moves]
    M = np.array([m if on_ee else inv_se3(m) for m in moves])   # the board in the view in frame f: inv(X) @ M[f] @ B

    # the start: each frame's board pose by PnP through view.K, then the hand-eye AX=XB over pairs of them
    boards = []
    for f in range(n):
        _, rvec, tvec = cv2.solvePnP(obj[f], img[f], view['K'], None)
        boards.append(rt_to_se3(cv2.Rodrigues(rvec)[0], tvec))

    def pairs_of(frames):
        pairs = [(i, j) for a, i in enumerate(frames) for j in frames[a + 1:]]
        if len(pairs) > HANDEYE_MAX_PAIRS:
            pairs = [pairs[p] for p in np.random.default_rng(0).choice(len(pairs), HANDEYE_MAX_PAIRS, replace=False)]
        return pairs
    if on_ee:                       # static camera, board on the gripper: X = the view in the base, B = the board
                                    # in the gripper
        start = list(range(n))
        for _ in range(OUTLIER_ROUNDS):     # a start without the frames that disagree far more than the rest
            pairs = pairs_of(start)
            X = solve_axxb([moves[j] @ inv_se3(moves[i]) for i, j in pairs],
                           [boards[j] @ inv_se3(boards[i]) for i, j in pairs])
            placed = np.array([(inv_se3(moves[i]) @ X @ boards[i])[:3, 3] for i in start])
            residual = np.linalg.norm(placed - np.median(placed, 0), axis=1)
            outliers = residual > OUTLIER_FACTOR * np.median(residual)
            if not outliers.any() or len(start) - outliers.sum() < MIN_VIEWS:
                break
            start = [i for i, out in zip(start, outliers) if not out]
        B = _mean_pose([inv_se3(moves[i]) @ X @ boards[i] for i in start])
    else:                           # moving camera, board fixed: X = the view in its parent, B = the board in the base
        pairs = pairs_of(list(range(n)))
        X = solve_axxb([inv_se3(moves[j]) @ moves[i] for i, j in pairs],
                       [boards[j] @ inv_se3(boards[i]) for i, j in pairs])
        B = _mean_pose([moves[i] @ X @ boards[i] for i in range(n)])

    K0 = view['K']
    k = np.array([K0[0, 0], K0[1, 1], K0[0, 2], K0[1, 2], 0.0, 0.0])
    xi, sigma, keep = np.zeros((n, 6)), np.array(ARM_ERROR_START), list(range(n))

    def fit(frames, k, X, B, sigma):
        return _bundle([obj[f] for f in frames], [img[f] for f in frames], M[frames], on_ee, k, X, B, xi[frames],
                       sigma)

    def pose_of(f, X, B, xi_f):     # frame f's board in the view, the arm's pose corrected
        E = rt_to_se3(_exp_so3(xi_f[None, :3])[0], xi_f[3:])
        return inv_se3(X) @ M[f] @ B @ E if on_ee else E @ inv_se3(X) @ M[f] @ B

    def residuals(f, k, X, B, xi_f):
        T = pose_of(f, X, B, xi_f)
        K, D = np.array([[k[0], 0.0, k[2]], [0.0, k[1], k[3]], [0.0, 0.0, 1.0]]), k[4:6]
        return _project(obj[f] @ T[:3, :3].T + T[:3, 3], K, D) - img[f]

    for round_ in range(OUTLIER_ROUNDS + 1):
        for _ in range(ARM_ROUNDS):         # the arm's accuracy, measured by the fit: the corrections' spread
            k, X, B, xs, var = fit(keep, k, X, B, sigma)
            xi[keep] = xs
            new = np.maximum(np.sqrt([np.mean(xs[:, :3] ** 2 + var[:, :3]), np.mean(xs[:, 3:] ** 2 + var[:, 3:])]),
                             ARM_ERROR_MIN)
            settled = bool(np.all(np.abs(new / sigma - 1.0) < 0.05))
            sigma = new
            if settled:
                break
        image = np.array([math.sqrt(np.mean(residuals(f, k, X, B, xi[f]) ** 2)) for f in range(n)])
        arm = np.sqrt(np.mean((xi / np.repeat(sigma, 3)) ** 2, axis=1))    # each correction, in the arm's errors
        bad = [f for f in keep if image[f] > OUTLIER_FACTOR * np.median(image[keep]) or arm[f] > OUTLIER_FACTOR]
        if not bad or len(keep) - len(bad) < MIN_VIEWS or round_ == OUTLIER_ROUNDS:
            break
        keep = [f for f in keep if f not in bad]
    rms = math.sqrt(np.mean(np.concatenate([residuals(f, k, X, B, xi[f]) for f in keep]) ** 2))
    rms_arm = math.sqrt(np.mean(np.concatenate([residuals(f, k, X, B, np.zeros(6)) for f in keep]) ** 2))

    # the uncertainty: test points across the view, from each refit's pose and intrinsics
    K, D = np.array([[k[0], 0.0, k[2]], [0.0, k[1], k[3]], [0.0, 0.0, 1.0]]), np.r_[k[4:6], 0.0, 0.0, 0.0]
    w, h = view['size']
    u, v = np.meshgrid(np.linspace(0, w - 1, 9), np.linspace(0, h - 1, 9))
    rays = (np.linalg.inv(K) @ np.stack([u.ravel(), v.ravel(), np.ones(u.size)])).T
    points = np.concatenate([rays * z for z in PREDICTION_DEPTHS_M['gripper' if on_ee else 'fixed']])
    points = points @ X[:3, :3].T + X[:3, 3]                          # in the frame the view pose X is in
    folds = min(UNCERTAINTY_FOLDS, len(keep))
    landed = []
    for g in range(folds):
        kg, Xg, *_ = fit([f for i, f in enumerate(keep) if i % folds != g], k, X, B, sigma)
        inv = inv_se3(Xg)
        Kg = np.array([[kg[0], 0.0, kg[2]], [0.0, kg[1], kg[3]], [0.0, 0.0, 1.0]])
        landed.append(_project(points @ inv[:3, :3].T + inv[:3, 3], Kg, kg[4:6]))
    landed = np.array(landed)
    spread = np.sqrt((folds - 1) / folds * ((landed - landed.mean(0)) ** 2).sum(axis=(0, 2)))
    placed = np.array([((inv_se3(moves[f]) if on_ee else moves[f]) @ X @ pose_of(f, X, B, xi[f]))[:3, 3] for f in keep])
    return {'K': K, 'D': D, 'X': X, 'B': B, 'keep': keep, 'rms': rms, 'rms_arm': rms_arm,
            'arm': (math.degrees(sigma[0]), 1000.0 * sigma[1]), 'uncertainty': float(np.percentile(spread, 95)),
            'folds': folds, 'placed': placed}


def _solve_view(cam, frames, good, objp, imgp, on_ee, board_spec, view, target=MAX_UNCERTAINTY_PX):
    """Calibrate the view as a camera of its own (_fit_view) from the frames ``good`` (their markers: objp, imgp), print
    what to review (a warning when the uncertainty is above ``target`` px), and return the camera's "calibration"
    entry."""
    robots = [np.asarray(frames[i]['ee_in_base'], float) for i in good]
    parents = [np.asarray(frames[i][f'parent__{cam}'], float) for i in good]
    fit = _fit_view(objp, imgp, robots if on_ee else parents, on_ee, view)
    K0, K, D, X, B, keep = view['K'], fit['K'], fit['D'], fit['X'], fit['B'], fit['keep']
    if len(keep) < len(good):
        print(f'  {cam}: {len(good) - len(keep)} frames dropped: their markers fit more than {OUTLIER_FACTOR}x worse '
              f'than the median frame, or the arm was off by more than {OUTLIER_FACTOR}x its error', flush=True)
    check = (f'board-in-{"gripper" if on_ee else "base"} translation std = '
             f'{1000 * np.linalg.norm(np.std(fit["placed"], axis=0)):.2f} mm')
    print(f'  {cam}: fitted over {len(keep)} frames: rms {fit["rms"]:.2f} px (view pixels), {fit["rms_arm"]:.2f} px '
          f'where the arm alone puts the board.  cross-check: {check}', flush=True)
    print(f'  {cam}: the arm\'s error per frame, as the fit measures it: {fit["arm"][0]:.2f} deg, {fit["arm"][1]:.1f} mm',
          flush=True)
    df = 100.0 * (np.array([K[0, 0], K[1, 1]]) / np.array([K0[0, 0], K0[1, 1]]) - 1.0)
    dc = K[:2, 2] - K0[:2, 2]
    print(f'  {cam}: the view camera: f=({K[0,0]:.1f},{K[1,1]:.1f}) c=({K[0,2]:.1f},{K[1,2]:.1f}) '
          f'k1={D[0]:+.4f} k2={D[1]:+.4f}; against view.K: f {df[0]:+.1f}% {df[1]:+.1f}%, c {dc[0]:+.1f} {dc[1]:+.1f} px',
          flush=True)
    print(f'  {cam}: uncertainty: where points across the view land varies by {fit["uncertainty"]:.1f} px (95th '
          f'percentile, {fit["folds"]} refits each without a share of the frames)', flush=True)
    if fit['uncertainty'] > target:
        print(f'  {cam}: WARNING the frames pin the view down poorly (> {target:g} px): jog again and let it run until '
              'it stops by itself', flush=True)
    if on_ee:
        size = board_spec['square_len_mm'] / 1000.0
        centre = B @ np.r_[board_spec['squares_x'] * size / 2, board_spec['squares_y'] * size / 2, 0.0, 1.0]
        print(f'  {cam}: the board on the gripper (ee_body frame): centre {np.round(1000 * centre[:3], 1).tolist()} '
              f'mm, facing {np.round(-B[:3, 2], 3).tolist()} -> "board_in_ee"', flush=True)
        axis, origin = X[:3, 2], X[:3, 3]                                  # X: the view in the base frame
        aim = ('its axis meets the base plane (z = 0) at '
               f'({origin[0] - origin[2] / axis[2] * axis[0]:.3f}, {origin[1] - origin[2] / axis[2] * axis[1]:.3f}) m'
               if axis[2] < -1e-6 else 'its axis points above the horizon')
    else:
        view_in_ee = inv_se3(inv_se3(parents[keep[0]]) @ robots[keep[0]]) @ X
        tilt = math.degrees(math.acos(np.clip(view_in_ee[2, 2], -1.0, 1.0)))
        aim = f'its axis is {tilt:.1f} deg from the ee_body z axis'
    print(f'  {cam}: the view: {aim}', flush=True)
    return {'K': np.round(K, 6).tolist(), 'D': np.round(D, 8).tolist(),
            'optical_in_parent': (inv_se3(parents[keep[0]]) @ X if on_ee else X).tolist(),
            'board_in_ee' if on_ee else 'board_in_base': np.round(B, 6).tolist(),
            'frames': len(keep), 'rms_px': round(fit['rms'], 3), 'arm_error_deg_mm': [round(v, 3) for v in fit['arm']],
            'uncertainty_px': round(fit['uncertainty'], 2), 'cross_check': check}


def _calibrate_camera(cam, work, on_ee, board_spec, raw, view=None, target=MAX_UNCERTAINTY_PX, others=()):
    """Calibrate ``cam``'s view (size, K, R; its frames reprojected through the ``raw`` lens): {'calibration'}. With no
    ``view``, fit the raw lens instead: {'raw'}."""
    frames = [np.load(p) for p in sorted(Path(work).glob('step_*.npz'))]
    if not frames:
        raise SystemExit(f'no frames (step_*.npz) in {work}')
    h, w = frames[0][f'raw__{cam}'].shape[:2]
    if view is not None and raw['size'] != [w, h]:
        raise SystemExit(f'its frames are {w}x{h}, but raw.size is {raw["size"]}: the lens was measured at another '
                         'resolution')
    good, objp, imgp = _detect(cam, work, frames, board_spec, raw, view, others)
    if view is None:
        K, D, rms = _fit_lens(cam, raw['lens'], (w, h), objp, imgp)
        return {'raw': {'lens': raw['lens'], 'size': [w, h], 'K': K.tolist(), 'D': D.ravel().tolist(),
                        'rms_px': round(rms, 4), 'views': len(good)}}
    return {'calibration': _solve_view(cam, frames, good, objp, imgp, on_ee, board_spec, view, target)}


def solve(cams, entries, meta, work, target=MAX_UNCERTAINTY_PX, others=None):
    """Calibrate each of ``cams`` from the capture in ``work`` (``entries``: its completed config entries; ``meta``: its
    meta.json): {camera: its entry with its "calibration"}, for the cameras that solved. One camera failing does not
    stop the others."""
    solved = {}
    for cam in cams:
        entry = entries[cam]
        on_ee, spec = entry['mount'] == 'gripper', entry['board']
        raw, view = _models(cam, entry, need_view=True)
        turned = math.degrees(math.acos(np.clip((np.trace(view['R']) - 1) / 2, -1.0, 1.0)))
        print(f'== {cam} ({"eye-to-hand, board on gripper" if on_ee else "eye-in-hand, board fixed"}), '
              f'board {spec["squares_x"]}x{spec["squares_y"]} @ {spec["square_len_mm"]}mm, {raw["lens"]} lens -> view '
              f'{view["size"][0]}x{view["size"][1]} f={view["K"][0, 0]:.1f}, turned {turned:.1f} deg from the camera '
              'axis ==', flush=True)
        try:
            result = _calibrate_camera(cam, work, on_ee, spec, raw, view, target, (others or {}).get(cam, ()))
        except (SystemExit, Exception) as e:               # one camera failing must not abort the others
            print(f'  {cam}: SKIPPED -- {e}', flush=True)
            continue
        out = json.loads(json.dumps(entry))                # the camera's completed entry, calibrated
        out['calibration'] = {**result['calibration'], 'calibrated': meta.get('captured')}
        solved[cam] = out
    return solved


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True,
                   help='the camera config (configs/droid_cameras.json): the NUC, the URDF, "ee_body", "table_z_m", and '
                        'each camera\'s serial, parent body, view, board and mount')
    p.add_argument('--serve-config', default='configs/droid_serve.json', help='the node graph, whose TCP is "ee_body"')
    p.add_argument('--config-out', help='write the config with each solved camera\'s completed "raw" and "view" and its '
                                        '"calibration" (only when every requested camera solved)')
    p.add_argument('--work-dir', required=True, help='raw capture stream (reused with --reuse-capture)')
    p.add_argument('--cameras', nargs='+',
                   help='cameras to calibrate (default: every config camera that defines a "board")')
    p.add_argument('--reuse-capture', action='store_true', help='solve an existing --work-dir; do not touch the robot')
    p.add_argument('--jog', action='store_true',
                   help='DRIVE the arm through random waypoints around where it stands, capturing each, until every '
                        "view's solve is good enough (stand at the e-stop; Ctrl-C ends it sooner)")
    p.add_argument('--amp', type=float, default=1.0, help='jog coverage amplitude scale (0-1) around the start')
    p.add_argument('--present-waist', action='store_true',
                   help='jog to PRESENT the gripper board to a camera near the arm\'s base (retract toward it) on half '
                        'of the waypoints')
    p.add_argument('--uncertainty-px', type=float, default=MAX_UNCERTAINTY_PX,
                   help='a capture goes on until each view\'s uncertainty (how far refits disagree on where points land, '
                        f'px) has been at most this for {JOG_STOP_FRAMES} board frames in a row; a solve above it warns '
                        '(default %(default)g)')
    p.add_argument('--interval', type=float, default=1.0, help='without --jog: seconds between frames')
    p.add_argument('--timeout', type=float, default=20.0, help='without --jog: end after this long without a frame')
    args = p.parse_args()

    config = json.loads(Path(args.config).read_text())
    cams = args.cameras or [c for c, v in config['cameras'].items() if 'board' in v]
    if not cams:
        raise SystemExit('nothing to calibrate: give a camera a "board" in the config, or pass --cameras')
    for cam in cams:
        c = config['cameras'].get(cam)
        if c is None:
            raise SystemExit(f'{cam} not in config cameras {list(config["cameras"])}')
        if 'board' not in c or c.get('mount') not in ('gripper', 'fixed'):
            raise SystemExit(f'{cam}: config needs a "board" spec and "mount" (gripper|fixed)')
    boards = []                                 # every board in the config, once: markers are told apart by fit
    for entry in config['cameras'].values():
        if 'board' in entry and entry['board'] not in boards:
            boards.append(entry['board'])
    others = {cam: [b for b in boards if b != config['cameras'][cam]['board']
                    and b['dict'] == config['cameras'][cam]['board']['dict']] for cam in cams}

    if not _have('cv2'):
        raise SystemExit('need opencv (cv2) to solve; run where cv2 is importable.')
    if args.jog and args.reuse_capture:
        raise SystemExit('--reuse-capture never touches the robot: drop --jog')
    work = Path(args.work_dir)
    if args.reuse_capture:
        try:
            entries = json.loads((work / 'cameras.json').read_text())
            meta = json.loads((work / 'meta.json').read_text())
        except FileNotFoundError:
            raise SystemExit(f'{work} holds no capture made by this script (cameras.json, meta.json)') from None
        missing = [c for c in cams if c not in entries]
        if missing:
            raise SystemExit(f'{work} holds no capture of {missing}')
    else:
        if not (_have('mujoco') and _have('zerorpc') and _have('pyzed')):
            raise SystemExit('capture needs mujoco, zerorpc and the ZED SDK\'s pyzed (README.md, Setup), or '
                             'pass --reuse-capture.')
        if 'nuc' not in config:
            raise SystemExit('capturing needs "nuc" (the NUC\'s control server) in the config')
        if args.jog and config.get('table_z_m') is None:
            raise SystemExit('--jog needs "table_z_m" in the config: the table\'s height in the arm base, measured with '
                             'the fingertips on it (scripts/arm.py where; README.md, Calibrate)')
        if any(work.glob('step_*.npz')):
            raise SystemExit(f'{work} already holds a capture: give a new --work-dir, or --reuse-capture to '
                             're-solve it')
        settings = json.loads((REPO / args.serve_config).read_text())
        try:
            entries, meta = capture(config, settings, cams, work, args.interval, args.timeout, args.uncertainty_px,
                                    jog=args.jog, amp=args.amp, present_waist=args.present_waist, others=others)
        except (RobotError, NotReady, OSError) as error:
            raise SystemExit(f'cannot reach the robot or a camera: {error}') from None

    solved = solve(cams, entries, meta, work, args.uncertainty_px, others)
    Path(work, 'calibration_results.json').write_text(json.dumps(solved, indent=2))
    failed = [c for c in cams if c not in solved]
    if args.config_out and failed:
        print(f'NOT writing {args.config_out}: {failed} did not solve; fix the capture and run again', flush=True)
    elif args.config_out:
        config['cameras'].update(solved)
        Path(args.config_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.config_out).write_text(json.dumps(config, indent=2) + '\n')
        print(f'wrote calibrated config -> {args.config_out}', flush=True)
    print('DONE. Review the rms + cross-check numbers before trusting the calibration.', flush=True)


if __name__ == '__main__':
    main()
