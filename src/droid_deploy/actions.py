"""A policy step -> joint position targets for the arm and the gripper (no robot I/O).

The step is the wire's two heads (rollout.proto StepRequest), as OmniAct's PolicyAdapter sends them, and it is
applied as omniact-molmospaces' eval_server applies it (embodiment_gen/representations.py ``SemanticAction.to_env``):
  action            (N,)       per-node RELATIVE joint targets (native rad / m) on the live joints, where action_mask is
                               set (joint nodes only)
  node_action_pose  (N, 3, 4)  the ABSOLUTE TCP pose (arm base <- WODS TCP axes), where node_action_pose_mask is set
                               (the TCP row only); the arm reaches it through the same IK (ik.py), seeded from the live
                               arm, with the joint head's arm target as its posture reference when that is masked too
                               ("hybrid"). Everything not commanded holds.
The gripper follows its driver nodes' targets. Targets are clipped to the URDF's joint range (``info['clipped']``).
The simulator bounds a step only by the IK's own 20 deg clamp; on the real arm a tick's target is sent at once, so with
``max_step_rad`` each arm joint's target is also kept within that of its measured position (``info['step_clamped']``).
The gripper is not limited: the NUC moves it at its own speed.
"""
from types import SimpleNamespace

import numpy as np

from .embodiment import JOINT_TCP, WODS_TCP
from .ik import IK_FAILURES, TcpIK


class Converter:
    """Shares the embodiment's kinematics. ``reset`` starts an episode (a fresh IK), ``convert`` turns one step into
    ``(start, target, info)`` over the controller's command joints (the arm joints, then the gripper joint)."""

    def __init__(self, embodiment, command_joints, max_step_rad=None):
        self.emb, self.kin = embodiment, embodiment.kin
        self.max_step = None if max_step_rad is None else float(max_step_rad)
        if self.max_step is not None and not self.max_step > 0:
            raise ValueError(f'max_step_rad must be positive (not {max_step_rad!r})')
        self.nodes, self.tcp_node = embodiment.nodes, embodiment.tcp_node
        self.joint_type, self.axis_mask = embodiment.joint_type, embodiment.axis_mask
        self.names = list(command_joints)
        drivers = [i for i, node in enumerate(self.nodes) if 'joint' in node and node['name'].startswith('gripper/')]
        self.arm_nodes = [i for i, node in enumerate(self.nodes) if 'joint' in node and i not in drivers]
        self.n_arm = n = len(self.arm_nodes)
        arm = [self.nodes[i]['joint'] for i in self.arm_nodes]
        model_order = [self.kin.model.joint(j).name for j in range(n)]
        if arm != self.names[:n] or arm != model_order or len(self.names) != n + 1:
            raise ValueError('the command joints must be the arm joints (the first model joints, as the IK expects) '
                             f'then one gripper joint: arm nodes {arm}, model {model_order}, command {self.names}')
        self.drivers = []                                        # (node, scale, offset): q_node = scale * q_gripper + offset
        for i in drivers:
            joint = self.nodes[i]['joint']
            leader, scale, offset = self.kin.mimic.get(joint, (joint, 1.0, 0.0))
            if leader != self.names[-1]:
                raise ValueError(f'gripper node {self.nodes[i]["name"]!r} is not driven by {self.names[-1]!r}')
            self.drivers.append((i, scale, offset))
        self.lower, self.upper = np.array([self.kin.limits(joint) for joint in self.names]).T
        base = np.linalg.inv(self.kin.reference_in_world())       # the wire frame, as the IK's base_pose
        quat = np.zeros(4)
        self.kin.mj.mju_mat2Quat(quat, np.ascontiguousarray(base[:3, :3]).reshape(9))
        self.base_pose = np.r_[base[:3, 3], quat]
        self.reset()

    def reset(self, seed=0):
        """A fresh IK for a new episode (it counts consecutive clamped steps; ``seed`` its retry perturbations)."""
        kin, n = self.kin, self.n_arm

        def pose(row):
            kin.data.qpos[:n] = row['arm']
            kin.data.qpos[n:] = row['gripper']
            kin.mj.mj_forward(kin.model, kin.data)
        self.ik = TcpIK(SimpleNamespace(mujoco=kin.mj, model=kin.model, data=kin.data, prefix='', n_arm=n, pose=pose),
                        seed=seed)

    def convert(self, joints, action, node_action_pose, action_mask, node_action_pose_mask):
        """Measured ``joints`` (name -> position) + one policy step -> ``(start, target, info)``: the command joints'
        measured positions, their targets, and what the conversion did."""
        action, pose, mask, pose_mask = self._validate(action, node_action_pose, action_mask, node_action_pose_mask)
        kin, n = self.kin, self.n_arm
        kin.set_joints(joints)
        qpos = kin.data.qpos.copy()
        start = np.array([joints[name] for name in self.names], float)
        current = np.array([kin.position(node['joint']) if 'joint' in node else 0.0 for node in self.nodes])
        wanted = current + np.where(mask, action, 0.0)           # absolute node targets from the joint head
        arm, gripper, info = wanted[self.arm_nodes], start[n], {}
        if mask[[i for i, _, _ in self.drivers]].any():
            gripper = float(np.mean([(wanted[i] - offset) / scale for i, scale, offset in self.drivers]))
        if pose_mask[self.tcp_node]:
            posture = arm if mask[self.arm_nodes].any() else None
            arm, ik = self.ik.command(pose[self.tcp_node], qpos[:n], qpos[n:], self.base_pose, posture=posture)
            info.update(ik)
            info['ik_failed'] = ik['ik_outcome'] in IK_FAILURES
        target = np.clip(np.r_[arm, gripper], self.lower, self.upper)
        clipped = [name for name, a, b in zip(self.names, np.r_[arm, gripper], target) if a != b]
        if clipped:
            info['clipped'] = clipped
        if self.max_step is not None:                            # around the measured joints, not the last command
            limited = np.clip(target[:n], start[:n] - self.max_step, start[:n] + self.max_step)
            stepped = [name for name, a, b in zip(self.names[:n], target[:n], limited) if a != b]
            if stepped:
                info['step_clamped'] = stepped
            target[:n] = limited
        return start, target, info

    def _validate(self, action, pose, action_mask, pose_mask):
        n = len(self.nodes)
        action, pose = np.asarray(action, float), np.asarray(pose, float)
        action_mask, pose_mask = np.asarray(action_mask), np.asarray(pose_mask)
        if action.shape != (n,) or pose.shape != (n, 3, 4) or action_mask.shape != (n,) or pose_mask.shape != (n,):
            raise ValueError(f'expected action ({n},), node_action_pose ({n}, 3, 4) and two ({n},) masks')
        if action_mask.dtype != bool or pose_mask.dtype != bool:
            raise ValueError('the masks must be bool')
        if not (np.isfinite(action).all() and np.isfinite(pose).all()):
            raise ValueError('the action is not finite')
        if (action_mask & ~self.axis_mask).any():
            raise ValueError('action_mask may select only joint nodes')
        if (pose_mask & (self.joint_type != JOINT_TCP)).any():
            raise ValueError('node_action_pose_mask may enable only the TCP row (joint type 3)')
        if not (action_mask.any() or pose_mask.any()):
            raise ValueError('the masks command nothing (action_mask and node_action_pose_mask are all False)')
        return action, pose, action_mask, pose_mask


def tcp_pose(embodiment, joints):
    """The TCP (WODS axes) in the arm base at measured ``joints``, 3x4: what node_action_pose's TCP row means."""
    embodiment.kin.set_joints(joints)
    return (embodiment.kin.site_pose() @ WODS_TCP)[:3]
