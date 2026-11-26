#!/usr/bin/env python3
"""The eval server: the real arm behind omniact-molmospaces' Rollout protocol, so OmniAct's eval client
(scripts/eval_omniact.py) drives the robot exactly as it drives the simulator's eval_server (README.md, Serve).

  .venv/bin/python scripts/serve.py --config configs/droid_cameras.json

It serves every calibrated camera's view and never judges success. Each session logs to out/serve/<time>/ and records
each episode as a WODS item (out/serve/wods/) and an annotated video. --no-motion runs the same loop with the arm
never commanded; --fake runs it on a simulated robot (no NUC, no cameras); --live-port also serves the live page
(README.md, Serve), which only this process can draw while it holds the cameras. Pressing r in its terminal moves the arm
to the start pose and opens the gripper, as a Reset does, while no client holds the robot.
"""
import argparse
import json
import os
import select
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np

from droid_deploy import REPO, motion
from droid_deploy.fake import FakeRobot
from droid_deploy.record import Recording
from droid_deploy.droid import RobotError
from droid_deploy.live import Live
from droid_deploy.live import serve as serve_live
from droid_deploy.server import RolloutService, serve
from droid_deploy.session import Session, move_to_start, open_options, setup


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True,
                   help='the camera config (configs/droid_cameras.json): the NUC, the arm\'s URDF, the cameras and '
                        'the DROID calibration file')
    p.add_argument('--serve-config', default='configs/droid_serve.json', help='node graph, start pose, tick, port')
    p.add_argument('--port', type=int, help='default: the serve config\'s (50051, as omniact-molmospaces\' eval_server)')
    p.add_argument('--no-motion', action='store_true',
                   help='never command the arm: each Step is converted and logged, not sent')
    p.add_argument('--live-port', type=int, help='also serve the live page of the views on this port')
    p.add_argument('--fake', action='store_true',
                   help='a simulated robot (off the robot); --config a calibrated config (calibrate_cameras.py --config-out)')
    args = p.parse_args()

    config = json.loads(Path(args.config).read_text())
    settings = json.loads((REPO / args.serve_config).read_text())
    port = args.port or int(settings['port'])
    try:
        if args.fake:
            cams = sorted(name for name, entry in config['cameras'].items() if 'calibration' in entry)
            start = np.r_[np.radians(settings['start_pose_deg']), float(settings['reset_gripper'])]
            robot = FakeRobot(config, cams, (REPO / config['robot_description']).read_text(),
                              dict(zip(config['motion']['command_joints'], start)))
        else:
            from droid_deploy.droid import Droid
            robot = Droid(config, command=not args.no_motion)
    except RobotError as error:
        raise SystemExit(str(error)) from None
    config = robot.config                                  # the camera entries completed: lens, view, calibration
    try:
        urdf = robot.wait_urdf()
        parts = setup(config, settings, urdf)
    except BaseException as error:
        robot.close()
        raise SystemExit(str(error)) from None
    log_dir = REPO / settings['log_dir']
    recording = Recording(log_dir / 'wods', config, settings, urdf)
    requester = 'serve'

    def open_session(bundle, options):
        prompt, views, horizon, continue_on_ik_failure, seed = open_options(bundle, options, parts.views,
                                                                            settings['task_horizon'])
        return Session(robot, parts.emb, parts.conv, settings, prompt=prompt, views=views, task_horizon=horizon,
                       continue_on_ik_failure=continue_on_ik_failure, seed=seed, motion_on=not args.no_motion,
                       requester=requester, log_dir=_session_dir(log_dir), recording=recording)

    service = RolloutService(open_session, settings['idle_timeout_s'])
    server = serve(service, port)
    live = Live(robot, parts.emb, parts.views) if args.live_port else None
    page = serve_live(live, args.live_port) if live else None
    print(f'serving the Rollout protocol on port {port}: views {parts.views}; '
          + ('fake robot' if args.fake else 'arm never commanded' if args.no_motion else 'motion on')
          + f'; logs in {log_dir}, episodes recorded to {recording.wods_root}'
          + (f'; live page on port {args.live_port}' if live else ''), flush=True)
    stop = threading.Event()
    for s in (signal.SIGINT, signal.SIGTERM):
        signal.signal(s, lambda *_: stop.set())

    def to_start():
        with motion.Controls(robot, requester):
            try:
                move_to_start(robot, settings, parts.conv.names, stop, requester)
            except motion.Stopped:
                motion.hold(robot, parts.conv.names)
                raise
    keys = ResetKey(service, to_start) if sys.stdin.isatty() and not args.no_motion else None
    try:
        stop.wait()
    finally:
        print('shutting down', flush=True)
        if keys:
            keys.close()
        service.shutdown()
        server.stop(2.0).wait()
        if live:
            page.shutdown()
            live.close()
        robot.close()


class ResetKey:
    """r in the server's terminal (read a key at a time, no Enter): the arm to the start pose, the gripper open, while
    no client holds the robot (one that does resets it with its own r)."""

    def __init__(self, service, to_start):
        import termios
        import tty
        self.service, self.to_start, self.fd = service, to_start, sys.stdin.fileno()
        self.saved, self.closed = termios.tcgetattr(self.fd), threading.Event()
        tty.setcbreak(self.fd)
        threading.Thread(target=self._run, daemon=True).start()
        print('press r here to move the arm to the start pose (while no client holds the robot)', flush=True)

    def _run(self):
        while not self.closed.is_set():
            if not select.select([self.fd], [], [], 0.2)[0] or os.read(self.fd, 1).lower() != b'r':
                continue
            print('r: the arm to the start pose', flush=True)
            try:
                self.service.outside_sessions(self.to_start)
                print('at the start pose', flush=True)
            except Exception as error:
                print(f'not reset: {error}', flush=True)

    def close(self):
        import termios
        self.closed.set()
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)


def _session_dir(root):
    """A new directory under ``root`` named by the time (two sessions in one second get -2, -3, ...)."""
    name = time.strftime('%Y%m%d-%H%M%S')
    for k in range(1, 1000):
        path = root / (name if k == 1 else f'{name}-{k}')
        try:
            path.mkdir(parents=True)
            return path
        except FileExistsError:
            continue
    raise RuntimeError(f'no free session directory for {name} under {root}')


if __name__ == '__main__':
    main()
