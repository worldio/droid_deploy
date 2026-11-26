# droid_deploy

Serve a real DROID (Franka Research 3, Robotiq 2F-85, ZED cameras) to [OmniAct](https://github.com/worldio/omniact)'s evaluation client over the Rollout gRPC protocol of [omniact-molmospaces](https://github.com/worldio/omniact-molmospaces), so the policy drives the arm as it drives the simulator.

<p align="center">
  <a href="https://arxiv.org/abs/2611.33158"><b>Paper</b></a> &nbsp;·&nbsp;
  <a href="https://omniact.github.io/"><b>Project page</b></a> &nbsp;·&nbsp;
  <a href="https://github.com/worldio/omniact"><b>Code</b></a>
</p>

## Setup

On the inference PC, wired to the NUC and with the ZED cameras plugged in:

```bash
git clone https://github.com/worldio/droid_deploy.git
cd droid_deploy
uv sync
uv pip install /path/to/pyzed-*-cp312-*.whl   # pyzed: the wheel the ZED SDK's get_python_api.py downloads
.venv/bin/python -c "import pyzed.sl as sl; print([d.serial_number for d in sl.Camera.get_device_list()])"
```

Put the printed serials into the cameras' `serial` fields in `configs/droid_cameras.json` (and `nuc`, if the NUC is not at the DROID default). After pulling changes, `uv sync --inexact` keeps `pyzed`.

## Calibrate

The NUC's DROID control server must be running, and nothing else may hold the robot or the cameras:

```bash
nc -zv <nuc> 4242
```

1. Print the two boards at 100% scale, mount the gripper board on the Robotiq, and fix the other board flat on the table under the gripper:

   ```bash
   .venv/bin/python scripts/calib/make_charuco.py --config configs/droid_cameras.json --camera exterior_1 --out-dir out/boards   # gripper board
   .venv/bin/python scripts/calib/make_charuco.py --config configs/droid_cameras.json --camera wrist --out-dir out/boards        # fixed board
   ```

2. Set the cameras' jog hints (each camera's `"jog": {...}` in `configs/droid_cameras.json`). Where a jog starts and which way it searches depend on where your cameras are, so the shipped config has none, and the defaults may not suit your station:

   - `"start_deg"`: the 7 arm joints (deg) the camera's jog starts and ends at, instead of the serve start pose. The exterior cameras usually need one: hand-guide the arm until the gripper board faces the camera from about 0.35–0.4 m, and paste the joints this prints:

     ```bash
     .venv/bin/python scripts/arm.py where --config configs/droid_cameras.json
     ```

   - `"search_dir"` (exterior cameras): the direction `[x, y, z]` in the arm's base frame (x forward, y to the robot's left, z up) in which the jog first carries the gripper board until the camera sees it; default `[1, 0, 1]`. Point it from the fingertips straight at the camera.
   - `"search_max_m"` (exterior cameras): how far that search goes, default 0.4 m; keep it short of the camera.

   Set them before step 3, or later in `out/droid_cameras_calibrated.json`, which the later commands read.

3. Calibrate the cameras in this order; each command reads the previous one's output (add `--amp 0.5` for a smaller first jog). Run each calibration until about 200 valid captures (the `board frames` count it prints); Ctrl-C ends it:

   ```bash
   # wrist: the gripper above the fixed board
   .venv/bin/python scripts/calibrate_cameras.py --jog --cameras wrist \
       --config configs/droid_cameras.json --config-out out/droid_cameras_calibrated.json --work-dir out/calib-wrist
   # exterior_1: from its jog.start_deg (step 2)
   .venv/bin/python scripts/calibrate_cameras.py --jog --cameras exterior_1 \
       --config out/droid_cameras_calibrated.json --config-out out/droid_cameras_calibrated.json --work-dir out/calib-ext1
   # exterior_2: likewise
   .venv/bin/python scripts/calibrate_cameras.py --jog --cameras exterior_2 \
       --config out/droid_cameras_calibrated.json --config-out out/droid_cameras_calibrated.json --work-dir out/calib-ext2
   ```

   To re-solve a capture without moving the arm, rerun its command with `--reuse-capture` in place of `--jog`.

4. Check the result at `http://<pc>:8080`: the node graph should lie on the arm in every view.

   ```bash
   .venv/bin/python scripts/visualize.py --config out/droid_cameras_calibrated.json
   ```

To use the DROID framework's own calibration instead, point `calibration_files` in `configs/droid_cameras.json` at its `droid/calibration/calibration_info.json` and serve with `--config configs/droid_cameras.json`.

## Serve

```bash
.venv/bin/python scripts/serve.py --config out/droid_cameras_calibrated.json --live-port 8080
.venv/bin/python scripts/serve.py --config out/droid_cameras_calibrated.json --no-motion   # never commands the arm
```

The server listens on port 50051. OmniAct's `scripts/eval_omniact.py` connects with `--target <pc>:50051` and picks the task with `--prompt`. If the GPU machine can't reach the PC, forward the port from the PC with `ssh -N -R 50061:127.0.0.1:50051 <user>@<gpu-machine>` and connect with `--target localhost:50061`.

## Citation

```bibtex
@misc{fan2026omniactpolicymillionembodiments,
      title={OmniAct: One Policy, One Million Embodiments},
      author={Xiang Fan and Yuheng Wang and Jiayuan Yan and Bohan Fang and Wisdom Oluchi Ikezogwo and Ainaz Eftekhar and Max Argus and Zhongzheng Ren and Jiafei Duan and Ranjay Krishna},
      year={2026},
      eprint={2611.33158},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2611.33158},
}
```
