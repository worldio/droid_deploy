"""The DROID: the arm and gripper through the DROID framework's control server on the NUC (zerorpc; its
scripts/server/run_server.py serving droid/franka/robot.py), the cameras through the ZED SDK. One thread owns the
connection to the NUC. It first starts the robot as the DROID framework's own client does on connecting
(droid/misc/server_interface.py): ``launch_controller`` restarts the NUC's robot and gripper processes, ``launch_robot``
connects the server to them; this moves nothing. Then it sends the newest command and reads the robot state,
``state_hz`` times a second. Each camera has a thread that keeps its newest left image, converted only when read.

All of it comes from the config (configs/droid_cameras.json): the NUC's address (``nuc``), the arm's URDF
(``robot_description``), each camera's ZED ``serial``, the DROID framework's calibration files
(``calibration_files``: each camera takes its newest calibration among them) and ``motion``. ``Droid.config`` is that config with each camera's entry completed (``complete``).
A camera whose entry already carries a ``calibration`` (scripts/calibrate_cameras.py's, in the config it writes) keeps
it instead of the DROID framework's: it holds only for the lens and view it was made through, so the ZED must report
that ``raw`` lens.

The gripper is one command joint, a Robotiq driver joint in rad. The NUC takes and reports it as a position from 0
(open) to 1 (closed), which ``motion.gripper_joint_range`` maps to the joint: its value at 0 and at 1.
"""
import copy
import json
import math
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from . import REPO

FRAME_MAX_AGE_S = 0.5    # a camera frame older than this is stale
JOINT_MAX_AGE_S = 0.2    # joint feedback older than this is stale: nothing moves on it
LENS_FOCAL_TOL = 0.005     # a camera's own calibration holds while the ZED's focal lengths stay within 0.5% of it
LENS_CENTRE_TOL_PX = 3.0   # ... and its principal point within 3 px (raw pixels)
NUC_TIMEOUT_S = 2.0      # each call to the NUC's control server (the tests shorten it)
LAUNCH_TIMEOUT_S = 60.0  # each of launch_controller (it waits 5 s for the processes) and launch_robot


class NotReady(RuntimeError):
    """Sensor data is missing or stale."""


class RobotError(RuntimeError):
    """The robot's software refused or did not answer (the URDF, a camera, the calibration file)."""


def complete(entry, lens=None, pose=None):
    """A camera's config entry as the eval server reads it (views.py). ``lens`` (K 3x3, D, (w, h): the ZED's left
    image, a pinhole) becomes ``raw``; the view's ``fov_deg`` becomes its K, a centred pinhole, and its R defaults to
    the camera's own axes; ``pose`` (the DROID calibration's [x, y, z, rx, ry, rz]: the camera in its parent body,
    OpenCV axes, Euler xyz) becomes ``calibration``. What is not given is left as the entry has it."""
    from scipy.spatial.transform import Rotation
    entry = copy.deepcopy(entry)
    if lens is not None:
        K, D, size = lens
        entry['raw'] = {'lens': 'pinhole', 'size': [int(v) for v in size], 'K': np.asarray(K, float).tolist(),
                        'D': np.asarray(D, float).ravel().tolist()}
    view = entry.setdefault('view', {})
    if 'fov_deg' in view:
        w, h = view['size']
        f = (w / 2.0) / math.tan(math.radians(float(view['fov_deg'])) / 2.0)
        view['K'] = [[f, 0.0, (w - 1) / 2.0], [0.0, f, (h - 1) / 2.0], [0.0, 0.0, 1.0]]
    view.setdefault('R', np.eye(3).tolist())
    if pose is not None:
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = Rotation.from_euler('xyz', pose[3:6]).as_matrix(), pose[:3]
        entry['calibration'] = {'optical_in_parent': T.tolist()}
    return entry


def _same_lens(name, entry, lens):
    """RobotError unless ``lens`` (K, D, (w, h): what the ZED reports, as Zed.lens) is the ``raw`` lens beside the entry's
    own calibration, the one it was made through."""
    raw = entry.get('raw')
    if not raw:
        raise RobotError(f'{name}: its calibration has no "raw" lens beside it: calibrate it again (README.md, Calibrate)')
    K, D, size = lens
    K, raw_K, D = np.asarray(K, float), np.asarray(raw['K'], float), np.asarray(D, float).ravel()
    # The ZED SDK self-calibrates each time a camera opens, so its rectified lens drifts a little between runs (0.6 px
    # of 522 seen): allow LENS_FOCAL_TOL of the focal lengths and LENS_CENTRE_TOL_PX of the centre, which moves a view
    # by well under a pixel. A different camera, resolution or lens model is far outside that.
    focal = np.abs(K[[0, 1], [0, 1]] / raw_K[[0, 1], [0, 1]] - 1.0).max()
    centre = np.abs(K[[0, 1], [2, 2]] - raw_K[[0, 1], [2, 2]]).max()
    same = ([int(v) for v in size] == [int(v) for v in raw['size']] and focal <= LENS_FOCAL_TOL
            and centre <= LENS_CENTRE_TOL_PX
            and len(D) >= len(raw['D']) and np.allclose(D[:len(raw['D'])], raw['D'], atol=1e-3))
    if same and (focal > 1e-4 or centre > 0.1):
        print(f'{name}: the ZED\'s lens differs a little from the one its calibration was made through '
              f'(f {K[0][0]:.1f} vs {raw_K[0][0]:.1f}, centre within {centre:.1f} px): within tolerance, used as calibrated',
              flush=True)
    if not same:
        raise RobotError(f'{name}: the ZED reports another lens ({int(size[0])}x{int(size[1])}, f={K[0][0]:.1f}) than the one '
                         f'its calibration was made through ({raw["size"][0]}x{raw["size"][1]}, f={raw["K"][0][0]:.1f}): '
                         'calibrate it again (README.md, Calibrate)')


def calibration_poses(paths):
    """The DROID framework's calibration files (droid/calibration/calibration_info.json, one per DROID copy): camera id
    (<serial>_left, <serial>_right) -> (pose, when it was calibrated, the file), the newest calibration of each camera
    among the files. An exterior camera's pose is in the arm base, the hand camera's in the flange."""
    poses = {}
    for path in [Path(p).expanduser() for p in paths]:
        if not path.is_file():
            raise RobotError(f'no DROID calibration file at {path}: correct calibration_files in the config')
        for cam_id, entry in json.loads(path.read_text()).items():
            stamp = float(entry.get('timestamp', 0.0))
            if cam_id not in poses or stamp > poses[cam_id][1]:
                poses[cam_id] = (entry['pose'], stamp, path)
    return poses


class Droid:
    """The NUC's control server and the ZED cameras of ``config``. With ``command`` it also sends the arm and gripper
    position targets. ``zed`` opens a camera by serial (default: the ZED SDK); ``cams`` names the cameras to open
    (default: every one with a serial)."""

    def __init__(self, config, command=False, zed=None, cams=None):
        motion = config['motion']
        self.names, self.command = list(motion['command_joints']), command
        self.setpoint_hz, self.state_hz = float(motion['setpoint_hz']), float(motion['state_hz'])
        self.gripper_speed = float(motion['gripper_speed'])
        self.gripper_open, self.gripper_closed = (float(v) for v in motion['gripper_joint_range'])
        self.address, self.urdf_path = config['nuc'], REPO / config['robot_description']
        self.lock, self._closing = threading.Lock(), threading.Event()
        self._joints = deque(maxlen=1024)        # (positions, stamp, arrival)
        self._frames, self._command, self._error, self._owner = {}, None, None, None
        self._launched = threading.Event()        # set when starting the robot on the NUC succeeded or gave up
        self.config = copy.deepcopy(config)
        poses = calibration_poses(config.get('calibration_files') or [])
        self._cameras, self._threads = {}, [threading.Thread(target=self._run_nuc, daemon=True)]
        try:
            for name, entry in config['cameras'].items():
                if not entry.get('serial') or (cams is not None and name not in cams):
                    continue
                camera = self._cameras[name] = (zed or Zed)(entry['serial'])
                own = entry.get('calibration')
                if own is not None:                  # calibrate_cameras.py's: kept, with the lens and view it holds for
                    _same_lens(name, entry, camera.lens)
                    self.config['cameras'][name] = complete(entry)
                    print(f"{name} (ZED {entry['serial']}): calibrated {own.get('calibrated') or '(when not recorded)'}"
                          f" by calibrate_cameras.py ({own.get('frames', '?')} frames, rms {own.get('rms_px', '?')} px)",
                          flush=True)
                else:
                    pose, stamp, path = poses.get(f"{entry['serial']}_left", (None, None, None))
                    self.config['cameras'][name] = complete(entry, camera.lens, pose)
                    print(f"{name} (ZED {entry['serial']}): " + (
                        f"calibrated {time.strftime('%Y-%m-%d %H:%M', time.localtime(stamp))} in {path} (DROID)"
                        if pose else 'no calibration: not served'), flush=True)
                self._threads.append(threading.Thread(target=self._run_camera, args=(name, camera), daemon=True))
        except BaseException:
            self.close()
            raise
        for thread in self._threads:
            thread.start()
        self._launched.wait()
        if self._error:
            self.close()
            raise RobotError(f'{self._error}. Is the NUC\'s control server running (launch_server.sh), the arm '
                             'unlocked and FCI on in Desk?')

    def _run_nuc(self):
        import gevent
        import zerorpc
        client = zerorpc.Client(heartbeat=20, timeout=NUC_TIMEOUT_S)
        client.connect(self.address)
        for attempt in (1, 2):                     # as the DROID framework's ServerInterface: twice
            try:
                client.launch_controller(timeout=LAUNCH_TIMEOUT_S)
                client.launch_robot(timeout=LAUNCH_TIMEOUT_S)
                break
            except Exception as error:
                self._error = f'starting the robot on the NUC at {self.address}: {type(error).__name__}: {error}'
                if attempt == 2:
                    self._launched.set()
                    client.close()
                    return
                gevent.sleep(0.1)
        self._error = None
        self._launched.set()
        span = self.gripper_closed - self.gripper_open
        while not self._closing.is_set():
            began = time.monotonic()
            try:
                with self.lock:
                    command, self._command = self._command, None
                if command is not None:
                    gripper = float(np.clip((command[-1] - self.gripper_open) / span, 0.0, 1.0))
                    client.update_command(command[:-1] + [gripper], 'joint_position', 'position', False)
                state, _ = client.get_robot_state()
                positions = dict(zip(self.names, state['joint_positions']))
                positions[self.names[-1]] = self.gripper_open + span * float(state['gripper_position'])
                with self.lock:
                    self._joints.append((positions, time.time(), time.monotonic()))
                    self._error = None
            except Exception as error:               # the state goes stale, so nothing moves; the status says why
                with self.lock:
                    self._error = f'the NUC at {self.address}: {type(error).__name__}: {error}'
            gevent.sleep(max(0.001, 1.0 / self.state_hz - (time.monotonic() - began)))
        client.close()

    def _run_camera(self, name, camera):
        while not self._closing.is_set():
            frame = camera.read()
            if frame is None:
                time.sleep(0.005)
                continue
            with self.lock:
                self._frames[name] = (*frame, time.monotonic())

    def wait_urdf(self, timeout=15.0):
        if not self.urdf_path.is_file():
            raise RobotError(f'no robot description at {self.urdf_path} (robot_description in the config)')
        return self.urdf_path.read_text()

    def joints(self, max_age=JOINT_MAX_AGE_S, wait=0.0):
        """The newest joint positions (name -> rad), at most ``max_age`` s old; waits up to ``wait`` s for them."""
        deadline = time.monotonic() + wait
        while True:
            with self.lock:
                latest = self._joints[-1] if self._joints else None
            if latest is not None and time.monotonic() - latest[2] <= max_age:
                return dict(latest[0])
            if time.monotonic() >= deadline:
                raise NotReady(self._error or 'no fresh joints from the NUC')
            time.sleep(0.05)

    def snapshot(self, cams, max_age=FRAME_MAX_AGE_S, max_skew=0.1):
        """(joints, frames): each camera's newest frame (RGB) and the joint sample closest to their capture time. NotReady
        if any of it arrived more than ``max_age`` s ago, or the stamps are more than ``max_skew`` s apart."""
        with self.lock:
            missing = [c for c in cams if c not in self._frames] + ([] if self._joints else ['joints'])
            if missing:
                raise NotReady(f'no data yet from {missing}' + (f' ({self._error})' if self._error else ''))
            frames = {c: self._frames[c] for c in cams}
            capture = float(np.mean([stamp for _, stamp, _ in frames.values()])) if frames else None
            joints = self._joints[-1] if capture is None else min(self._joints, key=lambda s: abs(s[1] - capture))
        now = time.monotonic()
        stale = [c for c, (_, _, t) in frames.items() if now - t > max_age] + (['joints'] if now - joints[2] > max_age else [])
        if stale:
            raise NotReady(f'stale data from {stale}')
        stamps = [stamp for _, stamp, _ in frames.values()] + [joints[1]]
        if max(stamps) - min(stamps) > max_skew:
            raise NotReady(f'camera and joint stamps are {max(stamps) - min(stamps):.3f} s apart')
        return dict(joints[0]), {c: _rgb(image) for c, (image, _, _) in frames.items()}

    def grab(self, cams, since=0.0, wait=0.0):
        """(frames, joints): each camera's newest frame (RGB), arrived after ``since`` (time.monotonic) and at most
        FRAME_MAX_AGE_S old, and the newest joints; waits up to ``wait`` s for them. NotReady names what is missing.
        The calibration's capture, once the arm has settled (scripts/calibrate_cameras.py)."""
        deadline = time.monotonic() + wait
        while True:
            now = time.monotonic()
            with self.lock:
                fresh = {c: self._frames[c] for c in cams
                         if c in self._frames and since < self._frames[c][2] and now - self._frames[c][2] <= FRAME_MAX_AGE_S}
            missing = [c for c in cams if c not in fresh]
            if not missing:
                joints = self.joints(JOINT_MAX_AGE_S, max(0.0, deadline - time.monotonic()))
                return {c: _rgb(image) for c, (image, _, _) in fresh.items()}, joints
            if now >= deadline:
                raise NotReady(f'no fresh frame from {missing}' + (f' ({self._error})' if self._error else ''))
            time.sleep(0.01)

    def status(self):
        """``homing`` (a Franka needs none), ``hardware`` ('active', or what went wrong with the NUC) and the controls
        ``owner``, the keys motion.py checks."""
        with self.lock:
            return {'homing': 'homed', 'hardware': self._error or 'active', 'owner': self._owner}

    def wait_status(self, key, timeout=3.0):
        return self.status()[key]

    def publish_command(self, positions):
        """The command joints' position targets (the arm in rad, then the gripper joint), sent by the NUC thread on
        its next turn: the newest one wins."""
        if not self.command:
            raise RobotError('this Droid was opened without commands')
        with self.lock:
            self._command = [float(x) for x in positions]

    def claim(self, requester):                      # nothing to claim: the NUC's server takes commands from any client
        self._owner = requester

    def release(self, requester):
        self._owner = None

    def close(self):
        self._closing.set()
        for thread in self._threads:
            if thread.is_alive():
                thread.join(timeout=2.0)
        for camera in self._cameras.values():
            camera.close()


ZED_W, ZED_H = 1280, 720          # HD720: the left image, BGRA
ZED_OPEN_TIMEOUT_S = 30.0


def _zed_worker(serial, shm_name, seq, stamps, lock, stop, pipe):
    """One ZED in a process of its own: pyzed holds the GIL through grab() (it never releases it), which in the server's
    process starved every other thread. Writes each left image into one of the two shared buffers, then bumps ``seq``:
    buffer ``seq % 2`` holds the newest frame, ``stamps[seq % 2]`` its capture time."""
    import os
    import signal
    from multiprocessing import shared_memory
    signal.signal(signal.SIGINT, signal.SIG_IGN)     # Ctrl-C belongs to the parent, which stops us through ``stop``
    parent = os.getppid()
    import pyzed.sl as sl
    camera, image, runtime = sl.Camera(), sl.Mat(), sl.RuntimeParameters()
    params = sl.InitParameters(camera_resolution=sl.RESOLUTION.HD720, camera_fps=60, depth_mode=sl.DEPTH_MODE.NONE)
    params.set_from_serial_number(int(serial))
    status = camera.open(params)
    if status != sl.ERROR_CODE.SUCCESS:
        pipe.send(('error', f'ZED {serial} did not open: {status} (is another program using it?)'))
        return
    try:
        configuration = camera.get_camera_information().camera_configuration
        left, size = configuration.calibration_parameters.left_cam, configuration.resolution
        if (size.width, size.height) != (ZED_W, ZED_H):
            pipe.send(('error', f'ZED {serial} streams {size.width}x{size.height}, not {ZED_W}x{ZED_H}'))
            return
        pipe.send(('ok', ([[left.fx, 0.0, left.cx], [0.0, left.fy, left.cy], [0.0, 0.0, 1.0]], list(left.disto)[:5],
                          (size.width, size.height))))
        shm = shared_memory.SharedMemory(name=shm_name)
        try:
            buffers = np.ndarray((2, ZED_H, ZED_W, 4), np.uint8, buffer=shm.buf)
            while not stop.is_set() and os.getppid() == parent:      # the parent gone: let the camera go
                if camera.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                    continue
                camera.retrieve_image(image, sl.VIEW.LEFT)
                stamp = camera.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds() * 1e-9
                k = (seq.value + 1) % 2                               # the buffer the reader is not on
                buffers[k] = image.get_data()
                stamps[k] = stamp
                with lock:
                    seq.value += 1
            del buffers
        finally:
            shm.close()
    finally:
        camera.close()


class Zed:
    """One ZED camera by serial number, its left image at the DROID framework's settings (HD720, 60 fps), read in a
    process of its own (_zed_worker) and handed over through shared memory, so the SDK never blocks this process."""

    def __init__(self, serial):
        import multiprocessing as mp
        from multiprocessing import shared_memory
        ctx = mp.get_context('spawn')                 # no fork: this process runs threads (gevent, gRPC, cameras)
        self.serial, self._last, self._proc = serial, 0, None
        self._shm = shared_memory.SharedMemory(create=True, size=2 * ZED_H * ZED_W * 4)
        self._buffers = np.ndarray((2, ZED_H, ZED_W, 4), np.uint8, buffer=self._shm.buf)
        self._seq, self._stamps = ctx.Value('q', 0, lock=False), ctx.Array('d', 2, lock=False)
        self._lock, self._stop = ctx.Lock(), ctx.Event()
        mine, theirs = ctx.Pipe(duplex=False)
        try:
            self._proc = ctx.Process(target=_zed_worker, name=f'zed-{serial}', daemon=True,
                                     args=(serial, self._shm.name, self._seq, self._stamps, self._lock, self._stop, theirs))
            self._proc.start()
            theirs.close()
            if not mine.poll(ZED_OPEN_TIMEOUT_S):
                raise RobotError(f'ZED {serial} did not open within {ZED_OPEN_TIMEOUT_S:.0f} s')
            try:
                kind, value = mine.recv()
            except EOFError:
                raise RobotError(f'ZED {serial}: its camera process ended while opening it '
                                 f'(exit code {self._proc.exitcode})') from None
            if kind != 'ok':
                raise RobotError(value)
            self.lens = value
        except BaseException:
            self.close()
            raise

    def read(self):
        """(the newest left image as the SDK gives it (BGRA), its capture time in s), or None if no new one came since
        the last read."""
        for _ in range(3):
            with self._lock:
                seq = self._seq.value
            if seq == self._last:
                if not self._proc.is_alive():
                    raise RobotError(f'ZED {self.serial}: its camera process ended (exit code {self._proc.exitcode})')
                return None
            k = seq % 2
            frame, stamp = self._buffers[k].copy(), float(self._stamps[k])
            with self._lock:
                unchanged = self._seq.value == seq             # else the worker may be writing buffer k now
            if unchanged:
                self._last = seq
                return frame, stamp
        return None

    def close(self):
        self._stop.set()
        if self._proc is not None and self._proc.pid is not None:
            self._proc.join(timeout=3.0)
            if self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(timeout=2.0)
        self._buffers = None
        try:
            self._shm.close()
            self._shm.unlink()
        except (BufferError, FileNotFoundError):
            pass


def _rgb(image):
    import cv2
    return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
