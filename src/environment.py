#!/usr/bin/env python

import rospy
import numpy as np
import math
from math import pi
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from nav_msgs.msg import OccupancyGrid
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from respawnGoal import Respawn

class Env():
    def __init__(self, action_size, alternative_rg_reward=False, alternative_rs_reward=False):
        self.goal_x = 0
        self.goal_y = 0
        self.heading = 0
        self.data_scan  = None
        self.data_costmap = None
        self.rate = rospy.Rate(4)
        self.action_size = action_size
        self.initGoal = True
        self.goal_distance =  float('Inf')
        self.last_goal_distance = float('Inf')
        self.alternative_rg_reward = alternative_rg_reward
        self.alternative_rs_reward = alternative_rs_reward
        self.position = Pose()
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry)
        self.sub_scan = rospy.Subscriber('scan', LaserScan, self.getScan)
        self.sub_costmap = rospy.Subscriber('/move_base/local_costmap/costmap', OccupancyGrid, self.getCostmap)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.respawn_goal = Respawn()

    def getPosition(self):
        return self.position.x, self.position.y

    def getGoal(self):
        goal_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)

        return self.goal_x, self.goal_y 

    def getGoalDistace(self):
        goal_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)

        return goal_distance

    def getScan(self, scan):
        self.data_scan = scan

    def getCostmap(self, costmap):
        self.data_costmap = costmap

    def getOdometry(self, odom):
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = euler_from_quaternion(orientation_list)

        goal_angle = math.atan2(self.goal_y - self.position.y, self.goal_x - self.position.x)

        heading = goal_angle - yaw
        if heading > pi:
            heading -= 2 * pi

        elif heading < -pi:
            heading += 2 * pi

        self.heading = round(heading, 2)

    def getState(self):
        try:
            self.rate.sleep()
        except:     
            pass   	
        while self.data_scan is None:
            pass
        while self.data_costmap is None:
            pass

        return self.data_scan, self.data_costmap

    def setReward(self, scan, action):
        rg = 0
        rc = 0
        rs = -5
        epsilon = 10

        min_range = 0.13
        scan_range = []

        if self.alternative_rs_reward:
            rs = 0

        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float('Inf'):
                scan_range.append(3.5)
            elif np.isnan(scan.ranges[i]):
                scan_range.append(0)
            else:
                scan_range.append(scan.ranges[i])

        if min_range > min(scan_range) > 0:
            rospy.loginfo("Collision!!")
            rc = -500
            if self.alternative_rg_reward:
                rc = -5000
            self.pub_cmd_vel.publish(Twist())
            return rg + rc + rs, True

        current_distance = self.getGoalDistace()
        if current_distance < 0.2:
            rospy.loginfo("Goal!!")
            rg = 500
            if self.alternative_rg_reward:
                rg = 5000
            self.pub_cmd_vel.publish(Twist())
            self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)
            return rg + rc + rs, True
        
        if not self.alternative_rg_reward:
            rg = epsilon * (self.last_goal_distance - current_distance)
        else:
            yaw_reward = []
            heading = self.heading # maybe changed since state created?

            for i in range(5):
                    angle = -pi / 4 + heading + (pi / 8 * i) + pi / 2
                    tr = 1 - 4 * math.fabs(0.5 - math.modf(0.25 + 0.5 * angle % (2 * math.pi) / math.pi)[0])
                    yaw_reward.append(tr)

            distance_rate = 2 ** (current_distance / self.goal_distance)
            r = ((round(yaw_reward[action] * 5, 2)) * distance_rate)                    
            rg = epsilon * r
        
        return rg + rc + rs, False
             
    def debug_step(self):
        data_scan, data_costmap = self.getState()           
        reward, done = self.setReward(data_scan, 2)
        self.last_goal_distance = self.getGoalDistace()

        return np.asarray(data_costmap.data), (self.goal_x - self.position.x, self.goal_y - self.position.y), (0, 0), reward, done        

    def step(self, action):
        max_angular_vel = 1.5
        lin_vel = 0.15
        ang_vel = ((self.action_size - 1)/2 - action) * max_angular_vel * 0.5

        rate = rospy.Rate(4)

        vel_cmd = Twist()
        vel_cmd.linear.x = lin_vel
        vel_cmd.angular.z = ang_vel
        self.pub_cmd_vel.publish(vel_cmd)

        # rate.sleep()

        # vel_cmd = Twist()
        # vel_cmd.linear.x = 0
        # vel_cmd.angular.z = 0
        # self.pub_cmd_vel.publish(vel_cmd)

        # rate.sleep()     


        data_scan, data_costmap = self.getState()           
        reward, done = self.setReward(data_scan, action)
        self.last_goal_distance = self.getGoalDistace()

        return np.asarray(data_costmap.data), (self.goal_x - self.position.x, self.goal_y - self.position.y), (lin_vel, ang_vel), reward, done

    def reset(self):
        rospy.loginfo("Reseting simulation...")
        rospy.wait_for_service('gazebo/reset_simulation')        
        try:
            self.reset_proxy()
            rospy.loginfo("      Done...")
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")

        _, data_costmap = self.getState()

        self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=not self.initGoal)
        self.goal_distance = self.getGoalDistace()
        self.last_goal_distance = self.getGoalDistace()
        self.initGoal = False

        return np.asarray(data_costmap.data), (self.goal_x - self.position.x, self.goal_y - self.position.y), (0.0, 0.0)
