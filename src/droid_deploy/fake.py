"""A simulated robot with the Droid interface (droid.py): the joints follow each command at once, the cameras send
noise frames at their lens's resolution, the arm is homed and the hardware active. serve.py --fake and visualize.py
--fake run on it, with a config whose camera entries are already complete (a calibrated config from calibrate_cameras.py --config-out)."""
import threading

import numpy as np


class FakeRobot:
    def __init__(self, config, cams, urdf_text, joints):
        self.config, self.urdf, self.names = config, urdf_text, list(config['motion']['command_joints'])
        self.setpoint_hz, self.gripper_speed = float(config['motion']['setpoint_hz']), float(config['motion']['gripper_speed'])
        self.sizes = {c: tuple(config['cameras'][c]['raw']['size']) for c in cams}
        self.q = dict(joints)                              # joint name -> position
        self.owner, self.commands, self.lock = None, 0, threading.Lock()
        self._rng = np.random.default_rng(0)

    def wait_urdf(self, timeout=15.0):
        return self.urdf

    def joints(self, max_age=0.2, wait=0.0):
        with self.lock:
            return dict(self.q)

    def snapshot(self, cams, max_age=0.5, max_skew=0.1):
        frames = {c: self._rng.integers(0, 255, (self.sizes[c][1], self.sizes[c][0], 3), np.uint8) for c in cams}
        return self.joints(), frames

    def grab(self, cams, since=0.0, wait=0.0):
        """(frames, joints), as Droid.grab: here every frame is new."""
        joints, frames = self.snapshot(cams)
        return frames, joints

    def status(self):
        return {'homing': 'homed', 'hardware': 'active', 'owner': self.owner}

    def wait_status(self, key, timeout=3.0):
        return self.status()[key]

    def publish_command(self, positions):
        with self.lock:
            self.q.update(zip(self.names, map(float, positions)))
            self.commands += 1

    def claim(self, requester):
        self.owner = requester

    def release(self, requester):
        self.owner = None

    def close(self):
        pass
