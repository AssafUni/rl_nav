#!/usr/bin/env python

import rospy
import os
import json
import numpy as np
import random
import time
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from collections import deque
from std_msgs.msg import Float32MultiArray
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
num_actions = 28
discount = 0.99

layers = tf.keras.layers

class DuelingDQN(tf.keras.Model):
  def __init__(self, num_actions):
    super(DuelingDQN, self).__init__()
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


loss_fn = tf.keras.losses.Huber()
optimizer = tf.keras.optimizers.Adam(lr=1e-5, clipnorm=10)

main_nn = None
target_nn = None

@tf.function
def train_step(costmaps, goal_vels, actions, rewards, next_costmaps, next_goal_vel, dones):
  next_qs_main = main_nn(next_costmaps, next_goal_vel)
  next_qs_argmax = tf.argmax(next_qs_main, axis=-1)
  next_action_mask = tf.one_hot(next_qs_argmax, num_actions)
  next_qs_target = target_nn(next_costmaps, next_goal_vel)
  masked_next_qs = tf.reduce_sum(next_action_mask * next_qs_target, axis=-1)
  target = rewards + (1. - dones) * discount * masked_next_qs
  with tf.GradientTape() as tape:
    qs = main_nn(costmaps, goal_vels)
    action_mask = tf.one_hot(actions, num_actions)
    masked_qs = tf.reduce_sum(action_mask * qs, axis=-1)
    loss = loss_fn(target, masked_qs)
  grads = tape.gradient(loss, main_nn.trainable_variables)
  optimizer.apply_gradients(zip(grads, main_nn.trainable_variables))
  return loss


class ReinforceAgent():
    def __init__(self, action_size, load_model=False, load_episode=0):
        self.pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
        self.dirPath = os.path.dirname(os.path.realpath(__file__))
        self.stage = rospy.get_param('/stage_number')
        stage_int = int(self.stage)
        if stage_int > 1:
            self.lastDirPath = self.dirPath.replace('rl_nav/src', 'rl_nav/save_model/stage_' + str(stage_int - 1) + "_")
        self.dirPath = self.dirPath.replace('rl_nav/src', 'rl_nav/save_model/stage_' + str(self.stage) + "_")
        self.result = Float32MultiArray()

        self.load_model = load_model or stage_int > 1
        self.load_episode = load_episode
        self.action_size = action_size
        self.episode_step = 300
        self.target_update = 100
        self.discount_factor = discount
        self.learning_rate = 5 * (10 ** -4)
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.05
        self.batch_size = 1024
        self.train_start = 1024
        self.memory = deque(maxlen= 2 * (10 ** 5))

        self.model = self.buildModel()
        self.target_model = self.buildModel()

        self.updateTargetModel()

        if self.load_model and self.load_episode > 0:
            path = self.dirPath + str(self.load_episode) + ".weights"
            with open(path, 'rb') as file:
                weights = pickle.load(file)   
            self.model.set_weights(weights)
        elif self.load_model:
            path = self.lastDirPath + "_last.weights"
            with open(path, 'rb') as file:
                weights = pickle.load(file)   
            self.model.set_weights(weights)            


    def preprocessCostmap(self, costmap):
        costmap_a = np.asarray(costmap).reshape((60, 60))
        costmap_b = np.asarray(costmap).reshape((60, 60))
        costmap_c = np.asarray(costmap).reshape((60, 60))
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

    def buildModel(self):
        return DuelingDQN(self.action_size)

    def getQvalue(self, reward, next_target, done):
        if done:
            return reward
        else:
            return reward + self.discount_factor * np.amax(next_target)

    def updateTargetModel(self):
        self.target_model.set_weights(self.model.get_weights())

    def getAction(self, costmap, goal_vel):
        if np.random.rand() <= self.epsilon:
            self.q_value = np.zeros(self.action_size)
            return random.randrange(self.action_size)
        else:
            self.q_value = self.model(costmap, goal_vel)
            return np.argmax(self.q_value)

    def appendMemory(self, costmap, goal_vel, action, reward,  next_costmap, next_goal_vel, done):
        self.memory.append((costmap, goal_vel, action, reward, next_costmap, next_goal_vel, done))

    def trainModel(self, target=False):
        global main_nn
        global target_nn
        mini_batch = random.sample(self.memory, self.batch_size)

        losses = []

        for i in range(self.batch_size):
            costmaps = mini_batch[i][0]
            goal_vel = mini_batch[i][1]
            actions = mini_batch[i][2]
            rewards = mini_batch[i][3]
            next_costmaps = mini_batch[i][4]
            next_goal_vel = mini_batch[i][5]
            dones = mini_batch[i][6]
     
            main_nn = self.model
            target_nn = self.target_model

            loss = train_step(costmaps, goal_vel, actions, rewards, next_costmaps, next_goal_vel, dones)
            losses.append(loss)
        
        return np.mean(losses)

if __name__ == '__main__':    
    rospy.init_node('turtlebot3_dqn')
    pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
    pub_get_action = rospy.Publisher('get_action', Float32MultiArray, queue_size=5)
    result = Float32MultiArray()
    get_action = Float32MultiArray()

    action_size = num_actions
    loss = float('inf')

    env = Env(action_size)

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
                if global_step <= agent.target_update:
                    loss = agent.trainModel()                  
                else:
                    loss = agent.trainModel(True)                   

            score += reward
            costmap, goal_vel = next_costmap, next_goal_vel
            get_action.data = [action, score, reward]
            pub_get_action.publish(get_action)

            if e % 100 == 0:
                last_weights_path = agent.dirPath + "_last.weights"
                weights_path = agent.dirPath + str(e) + ".weights"
                json_path = agent.dirPath + str(e) + '.json'
                with open(weights_path, 'wb') as outfile:
                    weights = agent.model.get_weights()
                    pickle.dump(weights, outfile)    
                with open(last_weights_path, 'wb') as outfile:
                    weights = agent.model.get_weights()
                    pickle.dump(weights, outfile)                                   
                with open(json_path, 'w') as outfile:
                    json.dump(param_dictionary, outfile)

            if t >= 500:
                rospy.loginfo("Time out!!")
                done = True

            if t % 10 == 0:
                m, s = divmod(int(time.time() - start_time), 60)
                h, m = divmod(m, 60)

                rospy.loginfo('Loss: %f, Ep: %d Step: %d score: %.2f memory: %d epsilon: %.2f time: %d:%02d:%02d',
                              loss, e, t, score, len(agent.memory), agent.epsilon, h, m, s)                

            if done:
                result.data = [score, np.max(agent.q_value)]
                pub_result.publish(result)
                agent.updateTargetModel()
                scores.append(score)
                episodes.append(e)
                m, s = divmod(int(time.time() - start_time), 60)
                h, m = divmod(m, 60)

                rospy.loginfo('Loss: %d, Ep: %d score: %.2f memory: %d epsilon: %.2f time: %d:%02d:%02d',
                              loss, e, score, len(agent.memory), agent.epsilon, h, m, s)
                param_keys = ['epsilon']
                param_values = [agent.epsilon]
                param_dictionary = dict(zip(param_keys, param_values))
                break

            global_step += 1
            if global_step % agent.target_update == 0:
                rospy.loginfo("UPDATE TARGET NETWORK")
              

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
