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

### This file is code the handles the interaction with the actual gazebo or real enviornment.
### It is also responsible to create the goal and delete it, and calculate the reward function.
### It uses both an own internal state, in addition to the agent state it returns to the dqn code.

# The class that handles the interaction with the actual gazebo or real enviornment.
class Env():
    # Initializes the enviornment, set action size to the number of possible actions of the agent
    def __init__(self, action_size):
        # Current goal position
        self.goal_x = 0
        self.goal_y = 0
        # Current heading of the robot, updated using a subscriber function
        self.heading = 0
        # Rate used to sleep when waiting for data
        self.rate = rospy.Rate(10)
        # Inital laser and costmaps data, updated using a subscriber function
        self.data_scan  = None
        self.data_costmap = None        
        self.action_size = action_size
        # Set to false after inital goal is created
        self.initGoal = True
        # inital goal distance and last step goal distance
        self.goal_distance =  float('Inf')
        self.last_goal_distance = float('Inf')
        self.position = Pose()
        # Creating neeeded publishers and subscribers to fetch data and interact with the enviornment, including a reset proxy
        # to reset the simulation
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry)
        self.sub_scan = rospy.Subscriber('scan', LaserScan, self.getScan)
        self.sub_costmap = rospy.Subscriber('/move_base/local_costmap/costmap', OccupancyGrid, self.getCostmap)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        # Used to create and delete the visual goal
        self.respawn_goal = Respawn()

    # Get robot current position
    def getPosition(self):
        return self.position.x, self.position.y

    # Get goal position
    def getGoal(self):
        return self.goal_x, self.goal_y 

    # Get goal distance
    def getGoalDistace(self):
        goal_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)

        return goal_distance

    # A callback function to update current laser scan
    def getScan(self, scan):
        self.data_scan = scan

    # A callback function to get current costmap
    def getCostmap(self, costmap):
        self.data_costmap = costmap        

    # A callback function to calculate current robot heading
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

    # A function to wait for new laser and costmap data, i.e. the internal enviornment state and the agent state
    def getState(self):
        try:
            self.rate.sleep()
        except:     
            pass   	
        while self.data_scan is None:
            try:
                self.rate.sleep()
            except:     
                pass  
        while self.data_costmap is None:
            try:
                self.rate.sleep()
            except:     
                pass  

        return self.data_scan, self.data_costmap, self.heading

    # Calculates the reward function, using a modified reward function from the paper
    def setReward(self, action, scan, heading):
        rg = 0 # Goal reward
        rc = 0 # Colission reward
        rs = -50 # Slow progress time reward
        epsilon = 10

        min_range = 0.13
        scan_range = []

        # Calculates current laser distances
        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float('Inf'):
                scan_range.append(3.5)
            elif np.isnan(scan.ranges[i]):
                scan_range.append(0)
            else:
                scan_range.append(scan.ranges[i])

        # If min laser range is smaller than min_range, robot is in a colission
        if min_range > min(scan_range) > 0:
            rospy.loginfo("Collision!!")
            rc = -5000
            self.pub_cmd_vel.publish(Twist())
            return rg + rc + rs, True

        # If current distance to goal is smaller than 0.2, we reached local goal, set reward,
        # reset simulation and create a new goal
        current_distance = self.getGoalDistace()
        if current_distance < 0.2:
            rospy.loginfo("Goal!!")
            rg = 5000
            self.pub_cmd_vel.publish(Twist())
            self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)
            self.goal_distance = self.getGoalDistace()
            return rg + rc + rs, True
        
        # Modified reward function than the paper using the original turtlebot3 code,
        # this function aids the agent because it adds immediate rewards if the robot is in 
        # the right heading and direction
        yaw_reward = []

        for i in range(5):
            angle = -pi / 4 + heading + (pi / 8 * i) + pi / 2
            tr = 1 - 4 * math.fabs(0.5 - math.modf(0.25 + 0.5 * angle % (2 * math.pi) / math.pi)[0])
            yaw_reward.append(tr)

        distance_rate = 2 ** (current_distance / self.goal_distance)
        r = ((round(yaw_reward[action] * 5, 2)) * distance_rate)                    
        rg = epsilon * r
        
        return rg + rc + rs, False
             
    # Next step of the agent
    def step(self, action):
        # Calculate linear and angular velocities for the action chosen
        max_angular_vel = 1.5
        lin_vel = 0.15
        ang_vel = ((self.action_size - 1)/2 - action) * max_angular_vel * 0.5

        #rate = rospy.Rate(4)

        # Sending action to actual enviornment
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

        # Getting current state
        data_scan, data_costmap, heading = self.getState()        
        # Calculating reward   
        reward, done = self.setReward(action, data_scan, heading)
        # Calculating last goal distance
        self.last_goal_distance = self.getGoalDistace()

        # Returning current state, reward and if the episode is done to the agent
        return np.asarray(data_costmap.data), (self.goal_x - self.position.x, self.goal_y - self.position.y), (lin_vel, ang_vel), heading, reward, done

    # Reseting the simulation
    def reset(self):
        # Reseting the simulation
        rospy.loginfo("Reseting simulation...")
        rospy.wait_for_service('gazebo/reset_simulation')        
        try:
            self.reset_proxy()
            rospy.loginfo("      Done...")
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")

        # Getting current state
        _, data_costmap, heading = self.getState()

        # Creating a new goal
        self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=not self.initGoal)
        self.goal_distance = self.getGoalDistace()
        self.last_goal_distance = self.getGoalDistace()
        self.initGoal = False

        # Returning current state
        return np.asarray(data_costmap.data), (self.goal_x - self.position.x, self.goal_y - self.position.y), (0.0, 0.0), heading
