#!/usr/bin/env python

import rospy
import time
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelState, ModelStates

class Moving():
    def __init__(self):
        self.pub_model = rospy.Publisher('gazebo/set_model_state', ModelState, queue_size=1)
        self.moving()

    def moving(self):
        while not rospy.is_shutdown():
            obstacle = ModelState()
            model = rospy.wait_for_message('gazebo/model_states', ModelStates)
            for i in range(len(model.name)):
                if model.name[i] == 'obstacle':
                    obstacle.model_name = 'obstacle'
                    obstacle.pose = model.pose[i]
                    obstacle.twist = Twist()
                    obstacle.twist.angular.z = 0.5
                    self.pub_model.publish(obstacle)
                    time.sleep(0.1)

def main():
    rospy.init_node('moving_obstacle')
    moving = Moving()

if __name__ == '__main__':
    main()