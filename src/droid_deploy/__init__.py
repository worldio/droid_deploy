"""Run OmniAct policies on a DROID (Franka arm, Robotiq gripper, ZED cameras): a Rollout server (serve.py) speaking
omniact-molmospaces' eval protocol. openyam_deploy's server, with droid.py where that has ros.py."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]      # the checkout: configs/ and out/ live here
