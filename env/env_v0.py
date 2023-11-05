import pdb
import random
import rospy
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState
import numpy as np
import math
from math import pi
import time
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3.common.env_checker import check_env

class Env(gym.Env):
    def __init__(self, args):
        super().__init__()
        self.namespace = args.namespace
        if args.continuous == 'False':
            self.action_space = spaces.Discrete(args.action_size)
            self.mode = 'discrete'
        else:
            self.action_space = spaces.Box(low=-1.,high=1.,shape=(2,),dtype=np.float32)
            self.mode = 'continuous'
        self.observation_space = spaces.Box(low=-np.inf,high=np.inf,shape=(362,),dtype=np.float32)
        self.goal_x = 0
        self.goal_y = 0
        self.status = 'initialized'
        self.map_x = -10.
        self.map_y = 0.
        self.rank = 0
        self.heading = 0
        self.action_size = args.action_size
        self.position = Pose()
        self.pub_cmd_vel = rospy.Publisher(self.namespace+'/cmd_vel', Twist, queue_size=5)
        self.sub_odom = rospy.Subscriber(self.namespace+'/odom', Odometry, self.getOdometry)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)

        # physics of tb3
        self.max_lin_vel = 0.15
        self.max_ang_vel = 1.5
        self.action_bound = np.array([0.15/2, 1.5])

        self.min_range = 0.13
        self.cur_step = 0
        self.steps_per_episode = 500
        self.cur_episode = 0
        self.rank_update_interval = args.rank_update_interval

    def getRelativeGoal(self):
        """
        Description: return relative goal distance and angle based on robot frame
        """
        # relative goal position based on robot base frame
        goal_pos = np.array([self.goal_x - self.position.x,
                             self.goal_y - self.position.y])

        goal_distance = np.linalg.norm(goal_pos)

        # relative goal position based on robot base frame
        rot = np.array([[np.cos(self.heading),np.sin(self.heading)],
                        [-np.sin(self.heading),np.cos(self.heading)]])  #(2,2)
        goal_pose = np.matmul(rot, goal_pos[:,None])[:,0]  #(2,)

        goal_angle = np.abs(np.arctan2(goal_pose[1],goal_pose[0]))

        # assert np.abs(goal_distance - np.linalg.norm(goal_pose)) < 1e-5, "goal pose distance should be kept same after rotation"

        return goal_distance, goal_angle, goal_pose

    def lidarPreprocess(self, scan):
        # remove inf/nan data from lidar scan
        scan_range = np.array(scan.ranges)
        scan_range[np.isnan(scan_range)] = 10.
        scan_range[np.isinf(scan_range)] = 10.

        return scan_range

    def getOdometry(self, odom):
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, self.heading = euler_from_quaternion(orientation_list)

    def getSarsa(self,scan):
        """
        Description:
        args:
            scan: list[360]
        return:
            state: np.array, (362,)
        """
        terminated = False
        reward = 0.

        goal_distance, goal_angle, goal_pose = self.getRelativeGoal()
        state = self.getState(scan, goal_pose)

        # goal distance reward
        if goal_distance < self.goal_distance:
            reward += self.goal_distance - goal_distance
        else:
            reward += 2 * (self.goal_distance - goal_distance)
        self.goal_distance = goal_distance

        # goal heading reward
        if goal_angle < self.goal_angle:
            reward += self.goal_angle - goal_angle
        else:
            reward += 2 * (self.goal_angle - goal_angle)
        self.goal_angle = goal_angle

        # check collision
        if self.min_range > np.min(state[:360]) > 0:
            terminated = True # done because of Collision
            reward = -200.
            self.pub_cmd_vel.publish(Twist())
            self.status = 'hit'

        # check if reached goal
        if goal_distance < 0.2:
            terminated = True
            reward = 200.
            self.pub_cmd_vel.publish(Twist())
            self.status = 'goal'

        return state, reward, terminated

    def getState(self, scan, goal_pose):
        """
        Description:
        args:
            scan: list[360]
            goal_pose: np.array(2,)
        return:
            state: np.array, (362,)
        """
        # pdb.set_trace()
        scan_range = self.lidarPreprocess(scan)

        return np.concatenate((scan_range, goal_pose), axis=0)

    def step(self, action):
        '''
        Description: env step() function
        args:
            action: (2,)
        return:
        '''
        if self.mode == 'continuous':
            action = (action + np.array([1.,0.])) * self.action_bound

            vel_cmd = Twist()
            vel_cmd.linear.x = action[0]
            vel_cmd.angular.z = action[1]
            self.pub_cmd_vel.publish(vel_cmd)
        else:
            max_angular_vel = 1.5
            ang_vel = ((self.action_size - 1)/2 - action) * max_angular_vel * 0.5

            vel_cmd = Twist()
            vel_cmd.linear.x = 0.15
            vel_cmd.angular.z = ang_vel
            self.pub_cmd_vel.publish(vel_cmd)

        # waiting more or less equal to 0.2 s until scan data received, stable behavior
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message(self.namespace+'/scan', LaserScan, timeout=5)
            except:
                pass

        state, reward, done = self.getSarsa(data)

        self.cur_step += 1
        truncated = False
        if self.cur_step >= self.steps_per_episode:
            truncated = True
            self.status = 'truncated'

        # print(f'cur_step:{self.cur_step}, '
        #       f'reward:{reward:.5f}, '
        #       f'done:{done}, '
        #       f'truncated:{truncated}')

        return np.asarray(state), reward, done, truncated, {"status": self.status}

    def collect_step(self, action):
        '''
        Description: env step() function
        args:
            action: (2,)
        return:
        '''
        vel_cmd = Twist()
        vel_cmd.linear.x = action[0]
        vel_cmd.angular.z = action[1]
        self.pub_cmd_vel.publish(vel_cmd)

        # waiting more or less equal to 0.2 s until scan data received, stable behavior
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message(self.namespace+'/scan', LaserScan, timeout=5)
            except:
                pass

        state, reward, done = self.getSarsa(data)

        self.cur_step += 1
        truncated = False
        if self.cur_step >= self.episode_step:
            truncated = True

        return np.asarray(state), reward, done, truncated, {"status": self.status}

    def render(self):
        pass

    def close(self):
        pass

    def reset(self, seed=None, options=None):
        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")

        # pdb.set_trace()
        self.reset_model() #reinitialize model starting position

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message(self.namespace+'/scan', LaserScan, timeout=5)
            except:
                pass

        self.status = 'initialized'
        self.goal_distance, self.goal_angle, goal_pose = self.getRelativeGoal()
        state = self.getState(data, goal_pose)

        self.cur_step = 0
        self.cur_episode += 1

        if self.cur_episode % self.rank_update_interval == 0:
            self.rank += 1

        return np.asarray(state), {"status": self.status}

    def reset_model(self):
        state_msg = ModelState()
        state_msg.model_name = self.namespace
        # update initial position and goal positions
        state_msg.pose.position.x, state_msg.pose.position.y, self.goal_x, self.goal_y = self.random_pts_map()
        self.init_x = state_msg.pose.position.x
        self.init_y = state_msg.pose.position.y

        # randomly initialize orientation
        yaw = np.pi * (random.random() * 2 - 1.)
        state_msg.pose.orientation.x = 0
        state_msg.pose.orientation.y = 0
        _, _, state_msg.pose.orientation.z, state_msg.pose.orientation.w = quaternion_from_euler(0.,0.,yaw)

        state_msg.pose.orientation.x = 0
        state_msg.pose.orientation.y = 0
        state_msg.pose.orientation.z = 0
        state_msg.pose.orientation.w = 1.

        # modify target cricket ball position
        target = ModelState()
        target.model_name = 'target_red'
        target.pose.position.x = self.goal_x
        target.pose.position.y = self.goal_y
        target.pose.position.z = 0.
        # target.pose.position.x = 0.
        # target.pose.position.y = 0.

        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            set_state( state_msg )
            set_state( target )

        except (rospy.ServiceException) as e:
            print("gazebo/set_model_state Service call failed")

    def random_pts_map(self):
        """
        Description: random initialize starting position and goal position, make distance > 0.1
        return:
            x1,y1: initial position
            x2,y2: goal position
        """
        # pdb.set_trace()
        # print(f'goal position is going to be reset')
        rx1,rx2 = np.random.uniform(0.16,1-0.16), np.random.uniform(0.16,1-0.16)
        ry1,ry2 = np.random.uniform(0.16,1-0.16), np.random.uniform(0.16,1-0.16)

        x1,x2,y1,y2 = rx1,rx2,ry1,ry2
        dist = np.linalg.norm([x1-x2,y1-y2])

        if self.rank == 0:
            while dist < 0.3 or dist > 4.:
                x1 = np.random.randint(5) + rx1
                x2 = np.random.randint(5) + rx2
                y1 = np.random.randint(5) + ry1
                y2 = np.random.randint(5) + ry2
                dist = np.linalg.norm([x1-x2,y1-y2])
        elif self.rank == 1:
            while dist < 0.3 or dist > 4.:
                x1 = np.array([2,1,1])[np.random.randint(3)] + rx1
                x2 = np.array([2,3,3])[np.random.randint(3)] + rx2
                y1 = np.array([1,1,2])[np.random.randint(3)] + ry1
                y2 = np.array([3,3,2])[np.random.randint(3)] + ry2
                dist = np.linalg.norm([x1-x2,y1-y2])
        elif self.rank == 2:
            while dist < 0.3 or dist > 4.:
                x1 = np.array([0,1,1])[np.random.randint(3)] + rx1
                x2 = np.array([3,3,4])[np.random.randint(3)] + rx2
                y1 = np.array([1,1,0])[np.random.randint(3)] + ry1
                y2 = np.array([4,3,3])[np.random.randint(3)] + ry2
                dist = np.linalg.norm([x1-x2,y1-y2])
        elif self.rank == 3:
            while dist < 0.3 or dist > 4.:
                x1 = np.array([0,1,1])[np.random.randint(3)] + rx1
                x2 = np.array([3,3,4])[np.random.randint(3)] + rx2
                y1 = np.array([1,1,0])[np.random.randint(3)] + ry1
                y2 = np.array([4,3,3])[np.random.randint(3)] + ry2
                dist = np.linalg.norm([x1-x2,y1-y2])

        x1 = self.map_x + self.rank * 5 + x1
        y1 = self.map_y + y1

        x2 = self.map_x + self.rank * 5 + x2
        y2 = self.map_y + y2

        return x1,y1,x2,y2
