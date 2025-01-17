import time
import numpy as np
import collections
import matplotlib.pyplot as plt
import dm_env

from constants import DT, START_ARM_POSE, MASTER_GRIPPER_JOINT_NORMALIZE_FN, PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN
from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN, PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN
from constants import PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE

from ml_bridge import MLBridge

import IPython
e = IPython.embed


class IsaacSimEnv:
    """
    Environment for real robot bi-manual manipulation
    Action space:      [left_arm_qpos (6),             # absolute joint position
                        left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
                        right_arm_qpos (6),            # absolute joint position
                        right_gripper_positions (1),]  # normalized gripper position (0: close, 1: open)

    Observation space: {"qpos": Concat[ left_arm_qpos (6),          # absolute joint position
                                        left_gripper_position (1),  # normalized gripper position (0: close, 1: open)
                                        right_arm_qpos (6),         # absolute joint position
                                        right_gripper_qpos (1)]     # normalized gripper position (0: close, 1: open)
                        "qvel": Concat[ left_arm_qvel (6),         # absolute joint velocity (rad)
                                        left_gripper_velocity (1),  # normalized gripper velocity (pos: opening, neg: closing)
                                        right_arm_qvel (6),         # absolute joint velocity (rad)
                                        right_gripper_qvel (1)]     # normalized gripper velocity (pos: opening, neg: closing)
                        "images": {"cam_high": (480x640x3),        # h, w, c, dtype='uint8'
                                   "cam_low": (480x640x3),         # h, w, c, dtype='uint8'
                                   "cam_left_wrist": (480x640x3),  # h, w, c, dtype='uint8'
                                   "cam_right_wrist": (480x640x3)} # h, w, c, dtype='uint8'
    """

    def __init__(self, setup_robots=True):
        self.bridge = MLBridge()
        self.bridge.open()
        self.qpos = np.zeros(14)
        self.sim_state = [0.0]

    def setup_robots(self):
        pass

    def get_qpos(self):
        return self.qpos

    def get_qvel(self):
        return np.zeros(14)

    def get_effort(self):
        return np.zeros(14)

    def get_images(self):
        observations, self.sim_state = self.bridge.read_observations()

        self.qpos = observations['qpos']
        return observations['images']

    def set_gripper_pose(self, left_gripper_desired_pos_normalized, right_gripper_desired_pos_normalized):
        # left_gripper_desired_joint = PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(left_gripper_desired_pos_normalized)
        # self.gripper_command.cmd = left_gripper_desired_joint
        # self.puppet_bot_left.gripper.core.pub_single.publish(self.gripper_command)

        # right_gripper_desired_joint = PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(right_gripper_desired_pos_normalized)
        # self.gripper_command.cmd = right_gripper_desired_joint
        # self.puppet_bot_right.gripper.core.pub_single.publish(self.gripper_command)
        pass

    def _reset_joints(self):
        # reset_position = START_ARM_POSE[:6]
        # move_arms([self.puppet_bot_left, self.puppet_bot_right], [reset_position, reset_position], move_time=1)
        pass

    def _reset_gripper(self):
        # """Set to position mode and do position resets: first open then close. Then change back to PWM mode"""
        # move_grippers([self.puppet_bot_left, self.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)
        # move_grippers([self.puppet_bot_left, self.puppet_bot_right], [PUPPET_GRIPPER_JOINT_CLOSE] * 2, move_time=1)
        pass

    def get_observation(self):
        obs = collections.OrderedDict()
        obs['images'] = self.get_images()
        obs['qpos'] = self.get_qpos()
        obs['qvel'] = self.get_qvel()
        obs['effort'] = self.get_effort()
        return obs

    def get_reward(self):
        return self.sim_state[0]

    def reset(self, fake=False):
        reward = self.bridge.reset_sim()
        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            # reward=self.get_reward(),
            reward=reward,
            discount=None,
            observation=self.get_observation())

    def step(self, action):
        state_len = int(len(action) / 2)
        left_action = action[:state_len]
        right_action = action[state_len:]

        self.bridge.write_joint_commands(np.concatenate((left_action, right_action)))

        time.sleep(DT)
        # this also reads in the reward so make sure to call it before self.get_reward()
        observation = self.get_observation()
        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=self.get_reward(),
            discount=None,
            observation=observation)
