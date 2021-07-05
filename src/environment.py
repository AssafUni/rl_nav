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
    def __init__(self, action_size):
        self.goal_x = 0
        self.goal_y = 0
        self.heading = 0
        self.action_size = action_size
        self.initGoal = True
        self.goal_distance =  float('Inf')
        self.position = Pose()
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.respawn_goal = Respawn()

    def getGoalDistace(self):
        goal_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)

        return goal_distance

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
        data_scan = None
        while data_scan is None:
            try:
                data_scan = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass
        data_costmap = None
        while data_costmap is None:
            try:
                data_costmap = rospy.wait_for_message('/move_base/local_costmap/costmap', OccupancyGrid, timeout=5)
            except:
                pass


        return data_scan, data_costmap

    def setReward(self, scan):
        rg = 0
        rc = 0
        rs = -5
        epsilon = 10

        min_range = 0.13
        scan_range = []

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
            self.pub_cmd_vel.publish(Twist())
            return rg + rc + rs, True

        current_distance = self.getGoalDistace()
        if current_distance < 0.2:
            rospy.loginfo("Goal!!")
            rg = 500
            self.pub_cmd_vel.publish(Twist())
            self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)
            return rg + rc + rs, False
        
        rg = epsilon * (self.goal_distance - current_distance)
        
        return rg + rc + rs, False
             
    def step(self, action):
        max_angular_vel = 1.5
        ang_vel = ((self.action_size - 1)/2 - action) * max_angular_vel * 0.5

        rate = rospy.Rate(4)

        vel_cmd = Twist()
        vel_cmd.linear.x = 0.15
        vel_cmd.angular.z = ang_vel
        self.pub_cmd_vel.publish(vel_cmd)

        rate.sleep()

        vel_cmd = Twist()
        vel_cmd.linear.x = 0
        vel_cmd.angular.z = 0
        self.pub_cmd_vel.publish(vel_cmd)

        rate.sleep()     


        data_scan, data_costmap = self.getState()           
        reward, done = self.setReward(data_scan)
        self.goal_distance = self.getGoalDistace()

        return np.asarray(data_costmap.data), (self.goal_x - self.position.x, self.goal_y - self.position.y), (vel_cmd.linear.x, vel_cmd.angular.z), reward, done

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
        self.initGoal = False

        return np.asarray(data_costmap.data), (self.goal_x - self.position.x, self.goal_y - self.position.y), (0.0, 0.0)