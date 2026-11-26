#!/usr/bin/env python3
"""Where the arm is, from the NUC, for setting up the camera calibration (README.md, Calibrate):

  .venv/bin/python scripts/arm.py where --config configs/droid_cameras.json

prints the arm's joints in degrees (ready to paste as a camera's "jog": {"start_deg": [...]}), the gripper joint, and
the fingertips (the node graph's TCP) in the arm base. With the fingertips resting on the table, their z is the table's
height in the base: the config's "table_z_m", which the calibration jog keeps the gripper above. It only reads, but
starting it restarts the robot processes on the NUC, as the server does (README.md, Serve); --fake reads the stand-in robot
at the start pose instead.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from droid_deploy import REPO
from droid_deploy.droid import JOINT_MAX_AGE_S, NotReady, RobotError
from droid_deploy.kinematics import Kinematics


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=['where'], help='where: the joints and the fingertips')
    p.add_argument('--config', required=True, help='the camera config (configs/droid_cameras.json)')
    p.add_argument('--serve-config', default='configs/droid_serve.json', help='the node graph, whose TCP is printed')
    p.add_argument('--fake', action='store_true', help='the stand-in robot at the start pose (off the robot)')
    args = p.parse_args()

    config = json.loads(Path(args.config).read_text())
    settings = json.loads((REPO / args.serve_config).read_text())
    urdf = (REPO / config['robot_description']).read_text()
    names = config['motion']['command_joints']
    if args.fake:
        from droid_deploy.fake import FakeRobot
        start = np.r_[np.radians(settings['start_pose_deg']), float(settings['reset_gripper'])]
        robot = FakeRobot(config, [], urdf, dict(zip(names, start)))
    else:
        from droid_deploy.droid import Droid
        try:
            robot = Droid(config, command=False, cams=[])                   # the NUC only: no camera is opened
        except RobotError as error:
            raise SystemExit(str(error)) from None
    try:
        joints = robot.joints(JOINT_MAX_AGE_S, wait=5.0)
    except NotReady as error:
        raise SystemExit(f'no joints from the NUC: {error}') from None
    finally:
        robot.close()

    tcp = next(n for n in settings['node_graph']['nodes'] if n.get('tcp'))
    kin = Kinematics(urdf, config['reference_body'], tcp['body'], tcp['site_rotation'], tcp['site_position'])
    kin.set_joints(joints)
    x, y, z = kin.site_pose()[:3, 3]
    arm = [round(float(np.degrees(joints[n])), 1) for n in names[:-1]]
    print(f'joints (deg): {arm}')
    print(f'  as a camera\'s jog start: "jog": {{"start_deg": {arm}}}')
    print(f'gripper ({names[-1]}): {joints[names[-1]]:.4f} rad')
    print(f'fingertips (the TCP) in {config["reference_body"]}: x {x:.4f}  y {y:.4f}  z {z:.4f} m'
          '   (resting on the table: z is "table_z_m")')


if __name__ == '__main__':
    main()
