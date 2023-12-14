import random
import pdb

import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
import numpy as np
import time
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import gymnasium as gym
from gymnasium import spaces

# import gym
# from gym import spaces
from stable_baselines3.common.env_checker import check_env

# for python 3.8
from typing import Tuple, List


class Env(gym.Env):
    def __init__(
        self, action_size, state_size, rank_update_interval=150, namespace="tb3"
    ):
        super().__init__()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # self.action_space = spaces.Discrete(action_size)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_size,), dtype=np.float32
        )
        self.goal_x = 0
        self.goal_y = 0
        self.namespace = namespace
        self.status = "initialized"

        # map info
        self.map_x = -10.0
        self.map_y = -10.0
        self.rank = 0

        # agent orientation quaternions with respect to world frame
        self.heading = []

        # agent 2d position axes with repect to word frame
        self.position = Pose()

        # publish action command data
        self.pub_cmd_vel = rospy.Publisher(
            self.namespace + "/cmd_vel", Twist, queue_size=5
        )
        # subscribe odometry data
        self.sub_odom = rospy.Subscriber(
            self.namespace + "/odom", Odometry, self.get_odometry
        )

        # reset proxy simulation platform
        self.reset_proxy = rospy.ServiceProxy("gazebo/reset_simulation", Empty)
        self.unpause_proxy = rospy.ServiceProxy("gazebo/unpause_physics", Empty)
        self.pause_proxy = rospy.ServiceProxy("gazebo/pause_physics", Empty)

        # physics of tb3
        self.max_lin_vel = 0.15
        self.max_ang_vel = 1.5
        self.action_bound = np.array([self.max_lin_vel / 2, self.max_ang_vel])

        self.min_range = 0.13
        self.cur_step = 0
        self.episode_step = 500

    def get_relative_goal(self) -> Tuple[np.float32, np.ndarray, np.ndarray]:
        """
        return relative goal position with respect to robot axes
        Returns:

        """
        # relative goal position based on robot base frame without rotation
        goal_tb_o_rot = np.array(
            [self.goal_x - self.position.x, self.goal_y - self.position.y]
        )

        goal_distance = np.linalg.norm(goal_tb_o_rot)

        _, _, heading = euler_from_quaternion([0.0, 0.0, *self.heading])

        # rotation matrix, (2,2)
        rot = np.array(
            [
                [np.cos(heading), np.sin(heading)],
                [-np.sin(heading), np.cos(heading)],
            ]
        )
        goal_tb_w_rot = np.matmul(rot, goal_tb_o_rot[:, None])[:, 0]  # (2,)

        goal_angle = np.abs(np.arctan2(goal_tb_w_rot[1], goal_tb_w_rot[0]))

        # assert np.abs(goal_distance - np.linalg.norm(goal_pose)) < 1e-5, "goal pose distance should be kept same after rotation"

        return goal_distance, goal_angle, goal_tb_w_rot

    def lidar_preprocess(self, scan: LaserScan) -> np.ndarray:
        """
        preprocess lidar raw data
        Args:
            scan: LaserScan(ranges:float, List[360])

        Returns:
            scan_range: np.float32, (360,)
        """
        # convert list of scan data to numpy array
        scan_range = np.array(scan.ranges)
        # set nan/inf data to nonsense data
        scan_range[np.isnan(scan_range)] = 10.0
        scan_range[np.isinf(scan_range)] = 10.0

        return scan_range

    def get_odometry(self, odom: Odometry) -> None:
        """
        subscribe odometry data and save it to self.heading(List[2])
        Args:
            odom: odom.pose + odom.orientation
        """
        self.position = odom.pose.pose.position
        self.heading = [odom.pose.pose.orientation.z, odom.pose.pose.orientation.w]

    def get_done(self, state: np.ndarray) -> Tuple[bool, bool]:
        """
        get terminated and truncated info based on current state data
        Args:
            state: np.float32, (360+2+2+2+2),(scan+position+heading+goal_pose)
        Returns:
           terminated: bool
        """
        terminated = False
        truncated = False

        # get goal distance
        goal_distance = np.linalg.norm(state[-2:])

        # check collision
        if 0.0 < np.min(state[:360]) < self.min_range:
            terminated = True
            self.pub_cmd_vel.publish(Twist())
            self.status = "hit"

        # check if reached goal
        if goal_distance < 0.2:
            terminated = True
            self.pub_cmd_vel.publish(Twist())
            self.status = "goal"

        # check if timed out
        self.cur_step += 1
        if self.cur_step >= self.episode_step:
            truncated = True
            terminated = True
            self.pub_cmd_vel.publish(Twist())
            self.status = "time out"

        return terminated, truncated

    def get_state(self, scan: LaserScan) -> np.ndarray:
        """
        return current state(scan + position + heading + goal)
        Args:
            scan: LaserScan
        Returns:
            state: np.float32, (360 + 2 + 2 + 2)
        """
        # get relative goal position with respect to robot frame
        _, _, goal_pose = self.get_relative_goal()

        # pdb.set_trace()
        scan_range = self.lidar_preprocess(scan)

        return np.concatenate(
            (scan_range, [self.position.x, self.position.y], self.heading, goal_pose),
            axis=0,
        )

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Description: env step() function
        args:
            action: np.float,(2,), range[0,1)
        return:
        """
        # publish action command
        action = (action + np.array([1.0, 0.0])) * self.action_bound

        # print(f"action: {action}")
        vel_cmd = Twist()
        vel_cmd.linear.x = action[0]
        vel_cmd.angular.z = action[1]
        self.pub_cmd_vel.publish(vel_cmd)

        # waiting more or less equal to 0.3s until scan data received, stable behavior
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message(
                    self.namespace + "/scan", LaserScan, timeout=5
                )
            except Exception as err:
                print(f"scan data not received: {err}")

        # get state
        state = self.get_state(data)
        # get reward
        reward = 0.0
        # get done and truncated info
        done, truncated = self.get_done(state)

        # print(f'cur_step:{self.cur_step}, '
        #       f'reward:{reward:.5f}, '
        #       f'done:{done}, '
        #       f'truncated:{truncated}')

        return state, reward, done, truncated, {"status": self.status}

    def render(self):
        pass

    def close(self):
        pass

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        """
        reset simulation and turtlebot3 model
        Args:
            seed ():
            options ():

        Returns:
            state, info
        """
        rospy.wait_for_service("gazebo/reset_simulation")
        try:
            self.reset_proxy()
        except rospy.ServiceException as e:
            print(f"gazebo/reset_simulation service call failed: {e}")

        # pdb.set_trace()
        self.reset_model()  # reinitialize model starting position

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message(
                    self.namespace + "/scan", LaserScan, timeout=5
                )
            except Exception as err:
                print(f"message not received: {err}")

        self.status = "running"

        state = self.get_state(data)

        # reset current step number
        self.cur_step = 0

        return state, {"status": self.status}

    def reset_model(self) -> None:
        """
        reset model position and orientation
        """
        state_msg = ModelState()
        state_msg.model_name = self.namespace

        # update initial position and goal positions
        (
            state_msg.pose.position.x,
            state_msg.pose.position.y,
            self.goal_x,
            self.goal_y,
        ) = self.random_pts_map()

        state_msg.pose.position.z = 0.0
        # randomly initialize orientation
        yaw = np.pi * (random.random() * 2 - 1.0)
        state_msg.pose.orientation.x = 0
        state_msg.pose.orientation.y = 0
        (
            _,
            _,
            state_msg.pose.orientation.z,
            state_msg.pose.orientation.w,
        ) = quaternion_from_euler(0.0, 0.0, yaw)

        state_msg.pose.orientation.x = 0
        state_msg.pose.orientation.y = 0

        rospy.wait_for_service("/gazebo/set_model_state")
        try:
            set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
            set_state(state_msg)

        except rospy.ServiceException as e:
            print(f"gazebo/set_model_state Service call failed: {e}")

    def random_pts_map(self, mode: str = "random") -> Tuple[float, float, float, float]:
        """
        Description: random initialize starting position and goal position, make distance > 0.1
        return:
            x1,y1: initial position
            x2,y2: goal position
        """
        # pdb.set_trace()
        # print(f'goal position is going to be reset')
        x1, x2, y1, y2 = 0, 0, 0, 0
        dist = np.linalg.norm([y2 - y1, x2 - x1])
        while dist < 0.3 or dist > 3.5:
            block_idx_1 = np.random.randint(5)
            block_idx_2 = np.random.randint(5)
            block_idy_1 = np.random.randint(5)
            block_idy_2 = np.random.randint(5)

            idx_1 = block_idy_1 * 5 + block_idx_1
            idx_2 = block_idy_2 * 5 + block_idx_2

            if idx_1 == 12 or idx_2 == 12:
                if self.rank in [1, 2, 3, 4, 8, 11]:
                    continue
            elif idx_1 == 6 or idx_2 == 6:
                if self.rank in [4, 5, 10, 12]:
                    continue
            elif idx_1 == 18 or idx_2 == 18:
                if self.rank in [4, 5, 12, 14]:
                    continue
            elif idx_1 == 7 or idx_2 == 7 or idx_1 == 17 or idx_2 == 17:
                if self.rank == 6:
                    continue
            elif idx_1 == 11 or idx_2 == 11 or idx_1 == 13 or idx_2 == 13:
                if self.rank == 7:
                    continue

            x1 = block_idx_1 + np.random.uniform(
                0.16, 1 - 0.16
            )  # random initialize x inside single map
            x2 = block_idx_2 + np.random.uniform(
                0.16, 1 - 0.16
            )  # random initialize goal position
            y1 = block_idy_1 + np.random.uniform(
                0.16, 1 - 0.16
            )  # random initialize y inside single map
            y2 = block_idy_2 + np.random.uniform(
                0.16, 1 - 0.16
            )  # random initialize goal position

            # update dist info
            dist = np.linalg.norm([y2 - y1, x2 - x1])

        if mode == "random":
            self.rank = random.randint(0, 15)

        x1 = self.map_x + (self.rank % 4) * 5 + x1
        y1 = self.map_y + ((15 - self.rank) // 4) * 5 + y1

        x2 = self.map_x + (self.rank % 4) * 5 + x2
        y2 = self.map_y + ((15 - self.rank) // 4) * 5 + y2

        return x1, y1, x2, y2


if __name__ == "__main__":
    env = Env(5, 362)
    check_env(env)
