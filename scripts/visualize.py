#!/usr/bin/env python3
"""A live web page of what the policy sees: every calibrated camera's view, reprojected as the server serves it, with
the node graph drawn on it from the arm's live joints, as omniact-molmospaces' visualize_node_graph.py draws a WODS frame
(README.md, Calibrate). It only listens to the robot (no commands). A ZED camera opens for one program at a time, so run
this while the server is down, to check the calibration; while serving, the server draws the same page itself
(serve.py --live-port).

  .venv/bin/python scripts/visualize.py --config configs/droid_cameras.json

then open http://<pc>:8080 (from elsewhere: ssh -N -L 8080:127.0.0.1:8080 <user>@<pc>). It draws only while a page is
open. --fake draws a simulated robot (no NUC, no cameras).
"""
import argparse
import json
import signal
import threading
from pathlib import Path

import numpy as np

from droid_deploy import REPO
from droid_deploy.droid import RobotError
from droid_deploy.embodiment import Embodiment
from droid_deploy.fake import FakeRobot
from droid_deploy.live import Live, serve


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True,
                   help='the camera config (configs/droid_cameras.json): the NUC, the arm\'s URDF, the cameras and '
                        'the DROID calibration file')
    p.add_argument('--serve-config', default='configs/droid_serve.json', help='the node graph')
    p.add_argument('--port', type=int, default=8080)
    p.add_argument('--host', default='0.0.0.0', help='the address to listen on (127.0.0.1: this machine only)')
    p.add_argument('--fps', type=float, default=10.0, help='the most drawings per second')
    p.add_argument('--scale', type=int, default=2, help='each view drawn at this many times its size')
    p.add_argument('--fake', action='store_true',
                   help='a simulated robot (off the robot); --config a calibrated config (calibrate_cameras.py --config-out)')
    args = p.parse_args()

    config = json.loads(Path(args.config).read_text())
    settings = json.loads((REPO / args.serve_config).read_text())
    try:
        if args.fake:
            start = np.r_[np.radians(settings['start_pose_deg']), float(settings['reset_gripper'])]
            robot = FakeRobot(config, sorted(config['cameras']), (REPO / config['robot_description']).read_text(),
                              dict(zip(config['motion']['command_joints'], start)))
        else:
            from droid_deploy.droid import Droid
            robot = Droid(config, command=False)
    except RobotError as error:
        raise SystemExit(str(error)) from None
    cameras = {name: entry for name, entry in robot.config['cameras'].items() if 'calibration' in entry}
    try:
        if not cameras:
            raise SystemExit('no calibrated camera: each camera in the config needs its ZED serial, and a pose under '
                             '<serial>_left in the DROID calibration file (README.md, Calibrate)')
        emb = Embodiment(settings['node_graph'], cameras, robot.wait_urdf(), config['reference_body'])
        live = Live(robot, emb, cameras, args.fps, args.scale)
        server = serve(live, args.port, args.host)
    except BaseException:
        robot.close()
        raise
    print(f'drawing {sorted(cameras)} on http://{args.host}:{args.port} (Ctrl-C stops)', flush=True)
    stop = threading.Event()
    for s in (signal.SIGINT, signal.SIGTERM):
        signal.signal(s, lambda *_: stop.set())
    try:
        stop.wait()
    finally:
        server.shutdown()
        live.close()
        robot.close()


if __name__ == '__main__':
    main()
