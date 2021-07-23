#!/usr/bin/env python

import rospy
import os
import json
import numpy as np
import random
import time
import sys
import math
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from collections import deque
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.srv import GetPlan, GetPlanRequest
from environment import Env
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.get_logger().setLevel('INFO')
import logging
tf.get_logger().setLevel(logging.ERROR)
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Dense, Dropout, Activation
import pickle


EPISODES = 3000
num_actions = 5
discount = 0.99
full_model = 0
goal_model = 1
goal_vel_model = 2

layers = tf.keras.layers

class DuelingDQNCostmapPlusGoalVelVector(tf.keras.Model):
  def __init__(self, num_actions):
    super(DuelingDQNCostmapPlusGoalVelVector, self).__init__()
    self.conv1 = layers.Conv2D(
        filters=32,
        kernel_size=8,
        strides=4,
        activation="relu",
        kernel_initializer=tf.keras.initializers.VarianceScaling(2.0),
        bias_initializer=tf.keras.initializers.Zeros(),
        # data_format="channels_first",
        padding='same'
    )    
    self.conv2 = layers.Conv2D(
        filters=64,
        kernel_size=4,
        strides=2,
        activation="relu",
        kernel_initializer=tf.keras.initializers.VarianceScaling(2.0),
        bias_initializer=tf.keras.initializers.Zeros(),
        # data_format="channels_first",
        padding='same'
    )
    self.conv3 = layers.Conv2D(
        filters=64,
        kernel_size=3,
        strides=1,
        activation="relu",
        kernel_initializer=tf.keras.initializers.VarianceScaling(2.0),
        bias_initializer=tf.keras.initializers.Zeros(),
        # data_format="channels_first",
        padding='same'
    )
    self.dense0 = layers.Dense(
        units=4,
        activation="relu",
        kernel_initializer=tf.keras.initializers.VarianceScaling(2.0),
        bias_initializer=tf.keras.initializers.Zeros(),
    )
    self.dense1 = layers.Dense(
        units=64,
        activation="relu",
        kernel_initializer=tf.keras.initializers.VarianceScaling(2.0),
        bias_initializer=tf.keras.initializers.Zeros(),
    )
    self.reshape = layers.Reshape((1, 1, 64))
    self.add = layers.Add()
    self.conv4 = layers.Conv2D(
        filters=64,
        kernel_size=3,
        strides=1,
        activation="relu",
        kernel_initializer=tf.keras.initializers.VarianceScaling(2.0),
        bias_initializer=tf.keras.initializers.Zeros(),
        # data_format="channels_first",
        padding='same'
    )    
    self.conv5 = layers.Conv2D(
        filters=64,
        kernel_size=3,
        strides=1,
        activation="relu",
        kernel_initializer=tf.keras.initializers.VarianceScaling(2.0),
        bias_initializer=tf.keras.initializers.Zeros(),
        # data_format="channels_first",
        padding='same'
    )   
    self.conv6 = layers.Conv2D(
        filters=64,
        kernel_size=3,
        strides=1,
        activation="relu",
        kernel_initializer=tf.keras.initializers.VarianceScaling(2.0),
        bias_initializer=tf.keras.initializers.Zeros(),
        # data_format="channels_first",
        padding='same'
    )  
    self.flatten = layers.Flatten()
    self.dense2 = layers.Dense(
        units=512,
        activation="relu",
        kernel_initializer=tf.keras.initializers.VarianceScaling(2.0),
        bias_initializer=tf.keras.initializers.Zeros(),
    )    
    self.dense3 = layers.Dense(
        units=512,
        activation="relu",
        kernel_initializer=tf.keras.initializers.VarianceScaling(2.0),
        bias_initializer=tf.keras.initializers.Zeros(),
    )
    self.V = layers.Dense(1)
    self.A = layers.Dense(num_actions)       

  @tf.function
  def call(self, costmaps, goal_vel_vector):
    x = self.conv1(costmaps)
    x = self.conv2(x)
    x = self.conv3(x)
    y = self.dense0(goal_vel_vector)
    y = self.dense1(y)
    y = self.reshape(y)
    o = tf.constant([1, 1, 8, 1], tf.int32) # int? not floats? velocities are floats?
    y = tf.tile(y, o)
    x = self.add([x, y])
    x = self.conv4(x)
    x = self.conv5(x)
    x = self.conv6(x)
    x = self.flatten(x)
    x = self.dense2(x)
    x = self.dense3(x)
    V = self.V(x)
    A = self.A(x)
    Q = V + tf.subtract(A, tf.reduce_mean(A, axis=1, keepdims=True))
    return Q

class DuelingDQNGoalVector(tf.keras.Model):
  def __init__(self, num_actions):
    super(DuelingDQNGoalVector, self).__init__()
    self.dense0 = layers.Dense(
        units=2,
        activation="relu",
        kernel_initializer=tf.keras.initializers.VarianceScaling(2.0),
        bias_initializer=tf.keras.initializers.Zeros(),
    )
    self.dense1 = layers.Dense(
        units=64,
        activation="relu",
        kernel_initializer=tf.keras.initializers.VarianceScaling(2.0),
        bias_initializer=tf.keras.initializers.Zeros(),
    )
    self.reshape = layers.Reshape((1, 1, 64))
    self.conv4 = layers.Conv2D(
        filters=64,
        kernel_size=3,
        strides=1,
        activation="relu",
        kernel_initializer=tf.keras.initializers.VarianceScaling(2.0),
        bias_initializer=tf.keras.initializers.Zeros(),
        # data_format="channels_first",
        padding='same'
    )    
    self.conv5 = layers.Conv2D(
        filters=64,
        kernel_size=3,
        strides=1,
        activation="relu",
        kernel_initializer=tf.keras.initializers.VarianceScaling(2.0),
        bias_initializer=tf.keras.initializers.Zeros(),
        # data_format="channels_first",
        padding='same'
    )   
    self.conv6 = layers.Conv2D(
        filters=64,
        kernel_size=3,
        strides=1,
        activation="relu",
        kernel_initializer=tf.keras.initializers.VarianceScaling(2.0),
        bias_initializer=tf.keras.initializers.Zeros(),
        # data_format="channels_first",
        padding='same'
    )  
    self.flatten = layers.Flatten()
    self.dense2 = layers.Dense(
        units=512,
        activation="relu",
        kernel_initializer=tf.keras.initializers.VarianceScaling(2.0),
        bias_initializer=tf.keras.initializers.Zeros(),
    )    
    self.dense3 = layers.Dense(
        units=512,
        activation="relu",
        kernel_initializer=tf.keras.initializers.VarianceScaling(2.0),
        bias_initializer=tf.keras.initializers.Zeros(),
    )
    self.V = layers.Dense(1)
    self.A = layers.Dense(num_actions)       

  @tf.function
  def call(self, goal_vector):
    y = self.dense0(goal_vector)
    y = self.dense1(y)
    y = self.reshape(y)
    o = tf.constant([1, 1, 8, 1], tf.int32) # int? not floats? velocities are floats?
    y = tf.tile(y, o)
    x = self.conv4(y)
    x = self.conv5(x)
    x = self.conv6(x)
    x = self.flatten(x)
    x = self.dense2(x)
    x = self.dense3(x)
    V = self.V(x)
    A = self.A(x)
    Q = V + tf.subtract(A, tf.reduce_mean(A, axis=1, keepdims=True))
    return Q

class DuelingDQNGoalVelVector(tf.keras.Model):
  def __init__(self, num_actions):
    super(DuelingDQNGoalVelVector, self).__init__()
    self.dense0 = layers.Dense(
        units=4,
        activation="relu",
        kernel_initializer=tf.keras.initializers.VarianceScaling(2.0),
        bias_initializer=tf.keras.initializers.Zeros(),
    )
    self.dense1 = layers.Dense(
        units=64,
        activation="relu",
        kernel_initializer=tf.keras.initializers.VarianceScaling(2.0),
        bias_initializer=tf.keras.initializers.Zeros(),
    )
    self.reshape = layers.Reshape((1, 1, 64))
    self.conv4 = layers.Conv2D(
        filters=64,
        kernel_size=3,
        strides=1,
        activation="relu",
        kernel_initializer=tf.keras.initializers.VarianceScaling(2.0),
        bias_initializer=tf.keras.initializers.Zeros(),
        # data_format="channels_first",
        padding='same'
    )    
    self.conv5 = layers.Conv2D(
        filters=64,
        kernel_size=3,
        strides=1,
        activation="relu",
        kernel_initializer=tf.keras.initializers.VarianceScaling(2.0),
        bias_initializer=tf.keras.initializers.Zeros(),
        # data_format="channels_first",
        padding='same'
    )   
    self.conv6 = layers.Conv2D(
        filters=64,
        kernel_size=3,
        strides=1,
        activation="relu",
        kernel_initializer=tf.keras.initializers.VarianceScaling(2.0),
        bias_initializer=tf.keras.initializers.Zeros(),
        # data_format="channels_first",
        padding='same'
    )  
    self.flatten = layers.Flatten()
    self.dense2 = layers.Dense(
        units=512,
        activation="relu",
        kernel_initializer=tf.keras.initializers.VarianceScaling(2.0),
        bias_initializer=tf.keras.initializers.Zeros(),
    )    
    self.dense3 = layers.Dense(
        units=512,
        activation="relu",
        kernel_initializer=tf.keras.initializers.VarianceScaling(2.0),
        bias_initializer=tf.keras.initializers.Zeros(),
    )
    self.V = layers.Dense(1)
    self.A = layers.Dense(num_actions)       

  @tf.function
  def call(self, goal_vel_vector):
    y = self.dense0(goal_vel_vector)
    y = self.dense1(y)
    y = self.reshape(y)
    o = tf.constant([1, 1, 8, 1], tf.int32) # int? not floats? velocities are floats?
    y = tf.tile(y, o)
    x = self.conv4(y)
    x = self.conv5(x)
    x = self.conv6(x)
    x = self.flatten(x)
    x = self.dense2(x)
    x = self.dense3(x)
    V = self.V(x)
    A = self.A(x)
    Q = V + tf.subtract(A, tf.reduce_mean(A, axis=1, keepdims=True))
    return Q


loss_fn = tf.keras.losses.Huber()
optimizer = tf.keras.optimizers.Adam(lr=1e-5, clipnorm=10)

main_nn = None
target_nn = None

@tf.function
def train_step_costmap_goal_vel_vector(costmap, goal_vel, action, reward, next_costmap, next_goal_vel, done):
  next_qs_main = main_nn(next_costmap, next_goal_vel)
  next_qs_argmax = tf.argmax(next_qs_main, axis=-1)
  next_action_mask = tf.one_hot(next_qs_argmax, num_actions)
  next_qs_target = target_nn(next_costmap, next_goal_vel)
  masked_next_qs = tf.reduce_sum(next_action_mask * next_qs_target, axis=-1)
  target = reward + (1. - done) * discount * masked_next_qs
  with tf.GradientTape() as tape:
    qs = main_nn(costmap, goal_vel)
    action_mask = tf.one_hot(action, num_actions)
    masked_qs = tf.reduce_sum(action_mask * qs, axis=-1)
    loss = loss_fn(target, masked_qs)
  grads = tape.gradient(loss, main_nn.trainable_variables)
  optimizer.apply_gradients(zip(grads, main_nn.trainable_variables))
  return loss

@tf.function
def train_step_goal_vector(goal, action, reward, next_goal, done):
  next_qs_main = main_nn(next_goal)
  next_qs_argmax = tf.argmax(next_qs_main, axis=-1)
  next_action_mask = tf.one_hot(next_qs_argmax, num_actions)
  next_qs_target = target_nn(next_goal)
  masked_next_qs = tf.reduce_sum(next_action_mask * next_qs_target, axis=-1)
  target = reward + (1. - done) * discount * masked_next_qs
  with tf.GradientTape() as tape:
    qs = main_nn(goal)
    action_mask = tf.one_hot(action, num_actions)
    masked_qs = tf.reduce_sum(action_mask * qs, axis=-1)
    loss = loss_fn(target, masked_qs)
  grads = tape.gradient(loss, main_nn.trainable_variables)
  optimizer.apply_gradients(zip(grads, main_nn.trainable_variables))
  return loss

@tf.function
def train_step_goal_vel_vector(goal_vel, action, reward, next_goal_vel, done):
  next_qs_main = main_nn(next_goal_vel)
  next_qs_argmax = tf.argmax(next_qs_main, axis=-1)
  next_action_mask = tf.one_hot(next_qs_argmax, num_actions)
  next_qs_target = target_nn(next_goal_vel)
  masked_next_qs = tf.reduce_sum(next_action_mask * next_qs_target, axis=-1)
  target = reward + (1. - done) * discount * masked_next_qs
  with tf.GradientTape() as tape:
    qs = main_nn(goal_vel)
    action_mask = tf.one_hot(action, num_actions)
    masked_qs = tf.reduce_sum(action_mask * qs, axis=-1)
    loss = loss_fn(target, masked_qs)
  grads = tape.gradient(loss, main_nn.trainable_variables)
  optimizer.apply_gradients(zip(grads, main_nn.trainable_variables))
  return loss


class ReinforceAgent():
    def __init__(self, action_size, load_params=True):
        self.pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
        self.dirPath = os.path.dirname(os.path.realpath(__file__))        
        load_model = False
        stage_int = 1
        load_episode = 0
        model_type = 0
        if load_params:
            self.stage = rospy.get_param('/stage_number')
            stage_int = int(self.stage)
            load_model = rospy.get_param('/load_model')
            load_model = int(load_model)
            load_model = load_model == 1
            load_episode = rospy.get_param('/load_episode')
            load_episode = int(load_episode)                
            if stage_int > 1:
                self.lastDirPath = self.dirPath.replace('rl_nav/src', 'rl_nav/save_model/stage_' + str(stage_int - 1) + "_")
            self.dirPath = self.dirPath.replace('rl_nav/src', 'rl_nav/save_model/stage_' + str(self.stage) + "_")
            model_type = rospy.get_param('/model_type')
            model_type = int(model_type)
        self.result = Float32MultiArray()

        self.model_type = model_type
        self.load_model = load_model or stage_int > 1
        self.load_episode = load_episode
        self.action_size = action_size
        self.episode_step = 1500
        self.discount_factor = discount
        self.learning_rate = 5 * (10 ** -4)
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.05
        self.batch_size = 64
        self.train_start = 64
        self.memory = deque(maxlen= 50000)
        self.costmapStep = 3
        self.costmapQueue = deque(maxlen=self.costmapStep * 3 + 1)
        
        self.model = self.buildModel(self.model_type)
        self.target_model = self.buildModel(self.model_type)

    def preprocessCostmap(self, costmap):
        self.costmapQueue.append(costmap)
        if len(self.costmapQueue) > self.costmapStep * 3:
            newQueue = deque(maxlen=10)            
            costmap_a = self.costmapQueue.pop()
            newQueue.append(costmap_a)
            counter = self.costmapStep
            while counter > 0:
                newQueue.append(self.costmapQueue.pop())
                counter = counter - 1
            costmap_b = self.costmapQueue.pop()
            newQueue.append(costmap_b)
            counter = self.costmapStep
            while counter > 0:
                newQueue.append(self.costmapQueue.pop())
                counter = counter - 1
            costmap_c =  self.costmapQueue.pop()
            newQueue.append(costmap_c)
            newQueue.reverse()
            self.costmapQueue = newQueue
        else:
            costmap_a = costmap
            costmap_b = costmap
            costmap_c = costmap
  
        costmap_a = np.asarray(costmap_a).reshape((60, 60))
        costmap_b = np.asarray(costmap_b).reshape((60, 60))
        costmap_c = np.asarray(costmap_c).reshape((60, 60))
        costmaps = np.zeros((3, 60, 60))
        costmaps[0] = costmap_a
        costmaps[1] = costmap_b
        costmaps[2] = costmap_c

        return np.expand_dims(costmaps, axis=0)

    def preprocessGoalAndVelocities(self, goal, velocities):
        goal_vector = np.asarray(goal)
        velocities_vector = np.asarray(velocities)
        goal_vel_vector = np.concatenate((goal_vector, velocities_vector), axis=None)
        return np.expand_dims(goal_vel_vector, axis=0)
    
    def extractGoalVector(self, goal_vel):        
        goal_vector = []
        goal_vector.append(goal_vel[0][0])
        goal_vector.append(goal_vel[0][1])
        goal_vector = np.asarray(goal_vector)
        return np.expand_dims(goal_vector, axis=0)

    def buildModel(self, model_type):
        if model_type == full_model:
            return DuelingDQNCostmapPlusGoalVelVector(self.action_size)
        elif model_type == goal_model:
            return DuelingDQNGoalVector(self.action_size)
        elif model_type == goal_vel_model:
            return DuelingDQNGoalVelVector(self.action_size)

    def updateTargetModel(self):
        self.target_model.set_weights(self.model.get_weights())

    def getAction(self, costmap, goal_vel):
        if np.random.rand() <= self.epsilon:
            self.q_value = np.zeros(self.action_size)
            return random.randrange(self.action_size)
        else:
            if self.model_type == full_model:
                self.q_value = self.model(costmap, goal_vel)
            elif self.model_type == goal_model:
                self.q_value = self.model(self.extractGoalVector(goal_vel))
            elif self.model_type == goal_vel_model:
                self.q_value = self.model(goal_vel)         
            
            return np.argmax(self.q_value)

    def do_train(self, costmap, goal_vel, action, reward, next_costmap, next_goal_vel, done):
        if self.model_type == full_model:
            train_step_costmap_goal_vel_vector(costmap, goal_vel, action, reward, next_costmap, next_goal_vel, done) 
        elif self.model_type == goal_model:
           train_step_goal_vector(self.extractGoalVector(goal_vel), action, reward, self.extractGoalVector(next_goal_vel), done) 
        elif self.model_type == goal_vel_model:
            train_step_goal_vel_vector(goal_vel, action, reward, next_goal_vel, done)          

    def appendMemory(self, costmap, goal_vel, action, reward,  next_costmap, next_goal_vel, done):
        self.memory.append((costmap, goal_vel, action, reward, next_costmap, next_goal_vel, done))

    def initNetwork(self):
        global main_nn
        global target_nn

        rospy.loginfo('Initializing network...')

        costmap = self.memory[0][0]
        goal_vel = self.memory[0][1]
        action = self.memory[0][2]
        reward = self.memory[0][3]
        next_costmap = self.memory[0][4]
        next_goal_vel = self.memory[0][5]
        done = self.memory[0][6]
    
        main_nn = self.model
        target_nn = self.target_model
         
        self.do_train(costmap, goal_vel, action, reward, next_costmap, next_goal_vel, done) 

        main_nn = self.target_model
        target_nn = self.model

        self.do_train(costmap, goal_vel, action, reward, next_costmap, next_goal_vel, done)             

        if self.load_model and self.load_episode > 0:
            rospy.loginfo('Loading saved model...')
            path = self.dirPath + str(self.load_episode) + ".weights"
            with open(path, 'rb') as file:
                weights = pickle.load(file)   
            self.model.set_weights(weights)
            self.target_model.set_weights(weights)

            with open(self.dirPath + str(self.load_episode) + '.json') as outfile:
                param = json.load(outfile)
                self.epsilon = param.get('epsilon')

	    self.load_model = False
        elif self.load_model:
            rospy.loginfo('Loading last saved model...')
            path = self.lastDirPath + "_last.weights"
            with open(path, 'rb') as file:
                weights = pickle.load(file)   
            self.model.set_weights(weights)   
            self.target_model.set_weights(weights) 

            with open(self.lastDirPath + "_last.json") as outfile:
                param = json.load(outfile)
                self.epsilon = param.get('epsilon')                   
		
	    self.load_model = False

    def trainModel(self):
        global main_nn
        global target_nn
        mini_batch = random.sample(self.memory, self.batch_size)

        losses = []

        for i in range(self.batch_size):
            costmap = mini_batch[i][0]
            goal_vel = mini_batch[i][1]
            action = mini_batch[i][2]
            reward = mini_batch[i][3]
            next_costmap = mini_batch[i][4]
            next_goal_vel = mini_batch[i][5]
            done = mini_batch[i][6]
     
            main_nn = self.model
            target_nn = self.target_model

            loss = self.do_train(costmap, goal_vel, action, reward, next_costmap, next_goal_vel, done) 
            losses.append(loss)
        
        return np.mean(losses)

class Agent():
    def __init__(self, action_size):
        self.r_a = ReinforceAgent(action_size, load_params=False)
        self.dirPath = os.path.dirname(os.path.realpath(__file__))
        self.stage = rospy.get_param('/stage_number')
        stage_int = int(self.stage)
        model_type = rospy.get_param('/model_type')
        self.model_type = int(model_type)        
        self.set_inital_pose(stage_int)
        self.dirPath = self.dirPath.replace('rl_nav/src', 'rl_nav/save_model/stage_' + str(self.stage) + "_")

        self.model = self.r_a.buildModel(model_type)

    def do_train(self, costmap, goal_vel, action, reward, next_costmap, next_goal_vel, done):
        if self.model_type == full_model:
            train_step_costmap_goal_vel_vector(costmap, goal_vel, action, reward, next_costmap, next_goal_vel, done) 
        elif self.model_type == goal_model:
           train_step_goal_vector(self.r_a.extractGoalVector(goal_vel), action, reward, self.extractGoalVector(next_goal_vel), done) 
        elif self.model_type == goal_vel_model:
            train_step_goal_vel_vector(goal_vel, action, reward, next_goal_vel, done) 

    def initNetwork(self, costmap, goal_dist, velocities):
        global main_nn
        global target_nn

        costmap = self.r_a.preprocessCostmap(costmap)
        goal_vel = self.r_a.preprocessGoalAndVelocities(goal_dist, velocities)

        rospy.loginfo('Initializing network...')

        main_nn = self.model
        target_nn = self.model

        self.do_train(costmap, goal_vel, 0, 0, costmap, goal_vel, False)            

        rospy.loginfo('Loading last saved model...')
        path = self.dirPath + "_last.weights"
        with open(path, 'rb') as file:
            weights = pickle.load(file)   
        self.model.set_weights(weights)   

    def getAction(self, costmap, goal_dist, velocities):
        costmap = self.r_a.preprocessCostmap(costmap)
        goal_vel = self.r_a.preprocessGoalAndVelocities(goal_dist, velocities)        
        self.q_value = self.model(costmap, goal_vel)
        return np.argmax(self.q_value)


    def set_inital_pose(self, stage):
        pub_pose = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=1)

        if stage == 1 or stage == 2 or stage == 3:
            p = PoseWithCovarianceStamped()
            p.header.stamp = rospy.Time.now()
            p.header.frame_id = "/map"
            p.pose.pose.position.x = 0.0107860857105
            p.pose.pose.position.y = 0.0073664161539
            p.pose.pose.position.z = 0.0
            p.pose.pose.orientation.x = 0.0
            p.pose.pose.orientation.y = 0.0
            p.pose.pose.orientation.z = -0.00327164459341
            p.pose.pose.orientation.w = 0.999994648157
            p.pose.covariance = [0.0020786870895943186, -0.00024191447863735466, 0.0, 0.0, 0.0, 0.0, -0.00024191447863735471, 0.001531162405457102, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001960197517311554]
        else:
            p = PoseWithCovarianceStamped()
            p.header.stamp = rospy.Time.now()
            p.header.frame_id = "/map"
            p.pose.pose.position.x = -0.70409776546
            p.pose.pose.position.y = 0.00666752422603
            p.pose.pose.position.z = 0.0
            p.pose.pose.orientation.x = 0.0
            p.pose.pose.orientation.y = 0.0
            p.pose.pose.orientation.z = -0.00327164459341
            p.pose.pose.orientation.w = 0.999994648157
            p.pose.covariance = [0.0020786870895943186, -0.00024191447863735466, 0.0, 0.0, 0.0, 0.0, -0.00024191447863735471, 0.001531162405457102, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001960197517311554]

        rate = rospy.Rate(4)
        while not rospy.is_shutdown():
            connections = pub_pose.get_num_connections()
            if connections > 0:
                for i in range(10):
                    pub_pose.publish(p)
                    return p.pose.pose.position.x, p.pose.pose.position.y
            rate.sleep()
        
        return None, None

    def take(self, array, m):
        local_goals = []
        for i in range(len(array)):
            if i % m == 0:
                local_goals.append(array[i])
        if len(array) - 1 % m != 0:
            local_goals.append(array[-1])
        return local_goals

    def global_plan(self, start_x, start_y, x, y, m):
        rospy.wait_for_service('/move_base/make_plan')
        make_plan_service = rospy.ServiceProxy('/move_base/make_plan', GetPlan)
        msg = GetPlanRequest()
        msg.start.header.frame_id = 'map'
        msg.start.pose.position.x = 0
        msg.start.pose.position.y = 0
        msg.start.pose.position.z = 0
        msg.start.pose.orientation.x = 0
        msg.start.pose.orientation.y = 0
        msg.start.pose.orientation.z = 0
        msg.start.pose.orientation.w = 0
        msg.goal.header.frame_id = 'map'
        msg.goal.pose.position.x = x
        msg.goal.pose.position.y = y
        msg.goal.pose.position.z = 0
        msg.goal.pose.orientation.x = 0
        msg.goal.pose.orientation.y = 0
        msg.goal.pose.orientation.z = 0
        msg.goal.pose.orientation.w = 0
        msg.tolerance = 2
        result = make_plan_service(msg) 
        return self.take(result.plan.poses, m)         
      

def debug_enviornment():
    alternative_rg_reward = rospy.get_param('/alternative_rg_reward')
    alternative_rg_reward = int(alternative_rg_reward)    
    alternative_rg_reward = alternative_rg_reward == 1
    alternative_rs_reward = rospy.get_param('/alternative_rs_reward')
    alternative_rs_reward = int(alternative_rs_reward)    
    alternative_rs_reward = alternative_rs_reward == 1  
    rate = rospy.Rate(1)
    action_size = num_actions
    env = Env(action_size, alternative_rg_reward, alternative_rs_reward)

    rospy.loginfo("Running...")
    _, goal_dist, _ = env.reset()
    while not rospy.is_shutdown():            
        _, goal_dist, _, reward, done = env.debug_step()
        rospy.loginfo("Goal dists: " + str(goal_dist))
        rospy.loginfo("Reward: " + str(reward))
        rospy.loginfo("Done: " + str(done))
        rate.sleep()


def train():
    pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
    pub_get_action = rospy.Publisher('get_action', Float32MultiArray, queue_size=5)
    result = Float32MultiArray()
    get_action = Float32MultiArray()

    action_size = num_actions
    loss = float('inf')

    alternative_rg_reward = rospy.get_param('/alternative_rg_reward')
    alternative_rg_reward = int(alternative_rg_reward)    
    alternative_rg_reward = alternative_rg_reward == 1
    alternative_rs_reward = rospy.get_param('/alternative_rs_reward')
    alternative_rs_reward = int(alternative_rs_reward)    
    alternative_rs_reward = alternative_rs_reward == 1  
    env = Env(action_size, alternative_rg_reward, alternative_rs_reward)

    agent = ReinforceAgent(action_size)
    scores, episodes = [], []
    global_step = 0
    start_time = time.time()

    last_100_ep_rewards = []
    for e in range(agent.load_episode + 1, EPISODES):
        done = False
        costmap, goal_dist, velocities = env.reset()
        costmap = agent.preprocessCostmap(costmap)
        goal_vel = agent.preprocessGoalAndVelocities(goal_dist, velocities)
        score = 0
        for t in range(agent.episode_step):            
            action = agent.getAction(costmap, goal_vel)

            next_costmap, next_goal_dist, next_velocities, reward, done = env.step(action)

            next_costmap = agent.preprocessCostmap(next_costmap)
            next_goal_vel = agent.preprocessGoalAndVelocities(next_goal_dist, next_velocities)

            agent.appendMemory(costmap, goal_vel, action, reward, next_costmap, next_goal_vel, done)

            if len(agent.memory) >= agent.train_start:                
                loss = agent.trainModel()                  
                                                  
            score += reward
            costmap, goal_vel = next_costmap, next_goal_vel
            get_action.data = [action, score, reward]
            pub_get_action.publish(get_action)

            if agent.load_model or (e == 1 and t == 1):
                agent.initNetwork()

                param_keys = ['epsilon']
                param_values = [agent.epsilon]
                param_dictionary = dict(zip(param_keys, param_values))
		
            if e % 10 == 0:
                last_weights_path = agent.dirPath + "_last.weights"
                weights_path = agent.dirPath + str(e) + ".weights"
                last_json_path = agent.dirPath + "_last.json"
                json_path = agent.dirPath + str(e) + '.json'
                with open(weights_path, 'wb') as outfile:
                    weights = agent.model.get_weights()
                    pickle.dump(weights, outfile)    
                with open(last_weights_path, 'wb') as outfile:
                    weights = agent.model.get_weights()
                    pickle.dump(weights, outfile)                                   
                with open(json_path, 'w') as outfile:
                    json.dump(param_dictionary, outfile)
                with open(last_json_path, 'w') as outfile:
                    json.dump(param_dictionary, outfile)

            if t >= agent.episode_step - 1:
                rospy.loginfo("Time out!!")
                done = True

            if t % 10 == 0:
                m, s = divmod(int(time.time() - start_time), 60)
                h, m = divmod(m, 60)

                rospy.loginfo('Loss: %f, Ep: %d Step: %d score: %.2f memory: %d epsilon: %.2f time: %d:%02d:%02d',
                              loss, e, t, score, len(agent.memory), agent.epsilon, h, m, s)                

            if done:
                rospy.loginfo("DONE! UPDATE TARGET NETWORK")
                result.data = [score, np.max(agent.q_value)]
                pub_result.publish(result)
                agent.updateTargetModel()
                scores.append(score)
                episodes.append(e)
                m, s = divmod(int(time.time() - start_time), 60)
                h, m = divmod(m, 60)

                rospy.loginfo('Loss: %f, Ep: %d score: %.2f memory: %d epsilon: %.2f time: %d:%02d:%02d',
                              loss, e, score, len(agent.memory), agent.epsilon, h, m, s)
                param_keys = ['epsilon']
                param_values = [agent.epsilon]
                param_dictionary = dict(zip(param_keys, param_values))
                break

            global_step += 1
            # if global_step % agent.target_update == 0:
            #     rospy.loginfo("UPDATE TARGET NETWORK")
              

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

def run():
    action_size = num_actions
    agent = Agent(action_size)
    env = Env(action_size)
    costmap, goal_dist, velocities = env.reset()
    agent.initNetwork(costmap, goal_dist, velocities)
    
    done = False
    rospy.loginfo("Running...")
    while not rospy.is_shutdown():
        action = agent.getAction(costmap, goal_dist, velocities)
        costmap, goal_dist, velocities, _, done = env.step(action)

        if done:
            rospy.loginfo("DONE!")
            done = False
            costmap, goal_dist, velocities = env.reset()

def prepareGlobalPlan(agent, env, plan_modulo_param):
    rospy.loginfo("Preparing global plan...")
    pos_x, pos_y = env.getPosition()
    global_goal_x, global_goal_y = env.getGoal()
    rospy.loginfo("\t From: " + str(pos_x) + ", " + str(pos_y))
    rospy.loginfo("\t To: " + str(global_goal_x) + ", " + str(global_goal_y))
    plan = agent.global_plan(pos_x, pos_y, global_goal_x, global_goal_y, plan_modulo_param)
    rospy.loginfo("-----------------------------------------------------------")
    rospy.loginfo("\t Got plan with: " + str(len(plan)) + " local goals...")
    return plan

def get_local_goal_dist(env, local_goal_x, local_goal_y):
    pos_x, pos_y = env.getPosition()

    return local_goal_x - pos_x, local_goal_y - pos_y

def get_local_goal_distance(env, local_goal_x, local_goal_y):
    pos_x, pos_y = env.getPosition()

    goal_distance = round(math.hypot(local_goal_x - pos_x, local_goal_y - pos_y), 2)

    return goal_distance

def follow_plan(env, plan, curr_index):
    if curr_index == -1:
        rospy.loginfo("Starting to follow plan...")
        curr_index = 0
    next_goal = plan[curr_index]
    local_goal_x, local_goal_y = next_goal.pose.position.x, next_goal.pose.position.y
    current_distance = get_local_goal_distance(env, local_goal_x, local_goal_y)
    if curr_index == 0:
        rospy.loginfo("Next local goal is: " + str(local_goal_x) + ", " + str(local_goal_y))
    if current_distance < 0.2:
        rospy.loginfo("Local goal reached...")
        curr_index = curr_index + 1
        if curr_index < len(plan):
            next_goal = plan[curr_index]
            local_goal_x, local_goal_y = next_goal.pose.position.x, next_goal.pose.position.y            
            rospy.loginfo("Next local goal is: " + str(local_goal_x) + ", " + str(local_goal_y))
            return curr_index
        else:
            rospy.loginfo("Global goal reached...")
            return -1
    return curr_index

def run_with_global_planner():
    plan_modulo_param = rospy.get_param('/plan_modulo')
    plan_modulo_param = int(plan_modulo_param)    
    action_size = num_actions
    agent = Agent(action_size)
    env = Env(action_size)
    costmap, _, velocities = env.reset()    
    plan = prepareGlobalPlan(agent, env, plan_modulo_param)
    next_goal = plan[0]
    local_goal_x, local_goal_y = next_goal.pose.position.x, next_goal.pose.position.y        
    goal_dist = get_local_goal_dist(env, local_goal_x, local_goal_y)    
    agent.initNetwork(costmap, goal_dist, velocities)
    
    done = False
    curr_index = -1
    rospy.loginfo("Running...")
    while not rospy.is_shutdown():
        curr_index = follow_plan(env, plan, curr_index)
        next_goal = plan[curr_index]
        local_goal_x, local_goal_y = next_goal.pose.position.x, next_goal.pose.position.y        
        goal_dist = get_local_goal_dist(env, local_goal_x, local_goal_y)
        action = agent.getAction(costmap, goal_dist, velocities)
        costmap, _, velocities, _, done = env.step(action)

        if curr_index == -1 and done:
            rospy.loginfo("DONE!")
            done = False
            costmap, _, velocities = env.reset()
            plan = prepareGlobalPlan(agent, env, plan_modulo_param)
        elif curr_index == -1:
            raise Exception("Invalid state curr_index == -1 without done+")    
        elif done:
            raise Exception("Invalid state done without curr_index == -1...")          


if __name__ == '__main__':    
    rospy.init_node('turtlebot3_dqn')
    train_param = rospy.get_param('/train')
    train_param = int(train_param)
    train_param = train_param == 1
    if train_param:
        rospy.loginfo("Running training...")
        train()
    else:
        run_type_param = rospy.get_param('/run_type')
        run_type_param = int(run_type_param)
        if run_type_param == 1:
            rospy.loginfo("Running trained agent with local goal...")
            run()
        elif run_type_param == 2:
            rospy.loginfo("Running trained agent with global planner and local goals...")
            run_with_global_planner()            
        elif run_type_param == 3:
            rospy.loginfo("Running enviornment debug...")
            debug_enviornment()
    
