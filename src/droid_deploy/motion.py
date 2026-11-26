"""Moving the arm: position setpoints sent to the NUC's control server (droid.py), ``robot.setpoint_hz`` a second
(the DROID framework's own rate is 15, one setpoint per policy tick).

``move`` is a point-to-point move (smoothstep, zero velocity at both ends, within a speed and acceleration limit):
the server's reset. ``stream`` plays one policy tick: a straight ramp from the last command to the new target over
the tick, so consecutive ticks join into one continuous motion. Both check before every setpoint (``check``): a stop
request, the NUC answering, fresh joint feedback and, given an ``owner``, the controls still ours. ``Controls`` holds
the controls for a block.
"""
import math
import time

import numpy as np

from .droid import JOINT_MAX_AGE_S, NotReady

SLEW_RAD_S = math.radians(45.0)      # move: speed limit
ACCEL_RAD_S2 = math.radians(90.0)    # move: acceleration limit (smoothstep: zero velocity at both ends)
MIN_MOVE_S = 0.4                     # move: shortest move


class Stopped(RuntimeError):
    """Motion stopped early: on request (Ctrl-C, SIGTERM, the client gone), the hardware paused or e-stopped, stale joint
    feedback, or the controls taken; the arm holds where it is."""


def check(robot, stop, owner=None):
    """Raise Stopped unless the motion may go on: no stop request (``stop``, a threading.Event), the hardware active,
    fresh joint feedback, and (``owner``) the controls still held by that requester."""
    if stop.is_set():
        raise Stopped('stop requested')
    status = robot.status()
    if status['hardware'] not in (None, 'active'):
        raise Stopped(f"hardware state is {status['hardware']!r}")
    if owner is not None and status.get('owner') not in (None, owner):
        raise Stopped(f"the controls owner is now {status['owner']!r}")
    try:
        robot.joints(JOINT_MAX_AGE_S)
    except NotReady as error:
        raise Stopped(str(error)) from None


def move(robot, start, target, stop, owner=None):
    """Send setpoints from ``start`` to ``target`` (command-joint order; the gripper last), smoothstep with zero
    velocity at both ends, within the speed and acceleration limits of the arm joints and the gripper's speed,
    checking before every setpoint. Returns ``target``."""
    start, target = np.asarray(start, float), np.asarray(target, float)
    span = float(np.abs(target - start)[:-1].max())
    duration = max(MIN_MOVE_S, 1.5 * span / SLEW_RAD_S, math.sqrt(6.0 * span / ACCEL_RAD_S2),
                   1.5 * abs(target[-1] - start[-1]) / robot.gripper_speed)
    n = max(1, int(math.ceil(duration * robot.setpoint_hz)))
    t0 = time.monotonic()
    for k in range(1, n + 1):
        check(robot, stop, owner)
        s = k / n
        robot.publish_command(start + (3 * s ** 2 - 2 * s ** 3) * (target - start))
        time.sleep(max(0.0, t0 + k / robot.setpoint_hz - time.monotonic()))   # on a clock: the checks' time doesn't add up
    return target


def stream(robot, start, target, duration, stop, owner=None):
    """Send setpoints on the straight line from ``start`` to ``target`` over ``duration`` s (one policy tick),
    checking before every setpoint. Returns ``target``."""
    start, target = np.asarray(start, float), np.asarray(target, float)
    n = max(1, int(round(duration * robot.setpoint_hz)))
    t0 = time.monotonic()
    for k in range(1, n + 1):
        check(robot, stop, owner)
        robot.publish_command(start + (k / n) * (target - start))
        time.sleep(max(0.0, t0 + k / robot.setpoint_hz - time.monotonic()))
    return target


def hold(robot, names):
    """Command the measured position (best effort)."""
    try:
        joints = robot.joints(JOINT_MAX_AGE_S)
        robot.publish_command([joints[n] for n in names])
    except Exception:
        pass


class Controls:
    """``with Controls(robot, requester):`` the controls, held for the block: the arm must be homed, and the claim
    is given back on exit."""

    def __init__(self, robot, requester):
        self.robot, self.requester = robot, requester

    def __enter__(self):
        homing = self.robot.wait_status('homing')
        if homing != 'homed':
            raise Stopped(f'the arm is not homed (reset status: {homing!r})')
        self.robot.claim(self.requester)
        return self

    def __exit__(self, *exc):
        self.robot.release(self.requester)
