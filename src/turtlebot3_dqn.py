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
# ------------ Setting No Verbose Logging- To see all warnings and other tensorflow logs, comment until *** -------------------
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
# **************************************
# ------------ Setting No Verbose Logging- To see all warnings and other tensorflow logs, comment until *** -------------------
tf.get_logger().setLevel('INFO')
import logging
tf.get_logger().setLevel(logging.ERROR)
# **************************************
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Dense, Dropout, Activation
import pickle

### This file is the main node of the Double Dueling DQN Reinforce agent.
### It creates the enviornment and agent, and runs the training loop. Parameters
### determine which stage to run, whether to load a saved model and more. See
### README for more info.

# The number of actions available to the agent, this will translate to angular velocities. See envrionment in getState.
# Modified from the paper
num_actions = 5
# The reward function discount factor.
discount = 0.99

layers = tf.keras.layers
tf.keras.backend.set_floatx('float32')

# The Dueling DQN keras/tensorflow model. See paper for more info.
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
        units=5,
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
  def call(self, costmaps, goal_vel_hed_vector):
    x = self.conv1(costmaps)
    x = self.conv2(x)
    x = self.conv3(x)
    y = self.dense0(goal_vel_hed_vector)
    y = self.dense1(y)
    y = self.reshape(y)
    o = tf.constant([1, 1, 8, 1], tf.int32)
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

# Double DQN main and target models
main_nn = None
target_nn = None

# The training step of the model, using both main and target models for double dqn
@tf.function
def train_step(costmap, goal_vel_hed, action, reward, next_costmap, next_goal_vel_hed, done):
  # Select best next action using main_nn.
  # Calculating q-action values using the next-costmap and next-goal-vel inputs
  next_qs_main = main_nn(next_costmap, next_goal_vel_hed)
  # Getting the action with the maximum value
  next_qs_argmax = tf.argmax(next_qs_main, axis=-1)
  # Creating a mask, i.e. one hot encoding of the max-action index
  next_action_mask = tf.one_hot(next_qs_argmax, num_actions)

  # Evaluate that best action using target_nn to know its Q-value.
  # Calculating q-action values using the next-costmap and next-goal-vel inputs
  next_qs_target = target_nn(next_costmap, next_goal_vel_hed)
  # Isolating the next-action value from the target model, but using the action index of the main model
  masked_next_qs = tf.reduce_sum(next_action_mask * next_qs_target, axis=-1)

  # Create target using the reward and the discounted next Q-value.
  # I.e, use the next q-value, or if the episode is done, only use the reward.
  target = reward + (1. - done) * discount * masked_next_qs
  with tf.GradientTape() as tape:
    # Q-values for the current state.
    # Calculating q-action values using the costmap and goal-vel input
    qs = main_nn(costmap, goal_vel_hed)
    # Creating a mask, i.e. one hot encoding of the actual action chosen index
    action_mask = tf.one_hot(action, num_actions)
    # Isolating the the action value
    masked_qs = tf.reduce_sum(action_mask * qs, axis=-1)
    # Calculating loss between q-value of the model to the target calculated
    loss = loss_fn(target, masked_qs)

  # Backpropagation
  grads = tape.gradient(loss, main_nn.trainable_variables)
  optimizer.apply_gradients(zip(grads, main_nn.trainable_variables))
  return loss


# The reinforce agent
class ReinforceAgent():
    # The init function of the agent. Set action_size to the amount of actions, and load_params whether or not to load stage_number, load_model and load_episode parameters.
    def __init__(self, action_size, load_params=True):
        # The directory of this file, will be used later to calculate where to save weights
        self.dirPath = os.path.dirname(os.path.realpath(__file__))
        load_model = False
        stage_int = 1
        load_episode = 0
        # Load parameters, stage number for the actual stage, load_model and load_episode to change whether to load a saved model and on which episode. See README for more info.
        if load_params:
            self.stage = rospy.get_param('/stage_number')
            stage_int = int(self.stage)
            load_model = rospy.get_param('/load_model')
            load_model = int(load_model)
            load_model = load_model == 1
            load_episode = rospy.get_param('/load_episode')
            load_episode = int(load_episode)                
            if stage_int > 1 and not load_model:
                self.lastDirPath = self.dirPath.replace('rl_nav/src', 'rl_nav/save_model/stage_' + str(stage_int - 1) + "_")
            self.dirPath = self.dirPath.replace('rl_nav/src', 'rl_nav/save_model/stage_' + str(self.stage) + "_")
        self.result = Float32MultiArray()

        # Setting up training parameters
        self.load_model = load_model or stage_int > 1
        self.load_episode = load_episode
        self.action_size = action_size
        if stage_int == 1:
            self.total_episodes = 250
        elif stage_int == 2:
            self.total_episodes = 20000
        elif stage_int == 3:
            self.total_episodes = 40000
        elif stage_int == 4:
            self.total_episodes = 80000 
        self.episode_step = 6000
        self.discount_factor = discount
        self.learning_rate = 5 * (10 ** -4)
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.05
        self.batch_size = 64
        self.train_start = 64
        self.memory = deque(maxlen=200000)
        self.costmapStep = 3
        self.costmapQueue = deque(maxlen=self.costmapStep * 3 + 1)
        
        # Creating target and main models
        self.model = self.buildModel()
        self.target_model = self.buildModel()

    # A function that takes a costmap, adds it to a queue, extracts costmaps
    # from the history and creates the input for the model which includes three costmaps for a 3-markov model
    def preprocessCostmap(self, costmap, reset_history=False):
        if reset_history:
            self.costmapQueue = deque(maxlen=self.costmapStep * 3 + 1)

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

    # Creates the goal-vel vector by creating a 4 dim vector
    def preprocessGoalAndVelocitiesAndHeading(self, goal, velocities, heading):
        goal_vector = np.asarray(goal)
        velocities_vector = np.asarray(velocities)
        heading_vector = np.asarray(heading)
        goal_vel_hed_vector = np.concatenate((goal_vector, velocities_vector, heading_vector), axis=None)
        return np.expand_dims(goal_vel_hed_vector, axis=0)

    # Creates the Double Dueling DQN model
    def buildModel(self):
        return DuelingDQN(self.action_size)

    # Updates the target model weights
    def updateTargetModel(self):
        self.target_model.set_weights(self.model.get_weights())

    # Calculates the next action using epsilon greedy approach, using the main model and a discounted over a preriod of time epsilon
    def getAction(self, costmap, goal_vel_hed):
        if np.random.rand() <= self.epsilon:
            self.q_value = np.zeros(self.action_size)
            return random.randrange(self.action_size)
        else:
            self.q_value = self.model(costmap, goal_vel_hed)
            return np.argmax(self.q_value)

    # Appends to the memory buffer the current states and next states
    def appendMemory(self, costmap, goal_vel_hed, action, reward,  next_costmap, next_goal_vel_hed, done):
        costmap = tf.convert_to_tensor(costmap, dtype=tf.float32)
        goal_vel_hed = tf.convert_to_tensor(goal_vel_hed, dtype=tf.float32)
        action = tf.convert_to_tensor(action, dtype=tf.int32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)
        next_costmap = tf.convert_to_tensor(next_costmap, dtype=tf.float32)
        next_goal_vel_hed = tf.convert_to_tensor(next_goal_vel_hed, dtype=tf.float32)
        done = tf.convert_to_tensor(done, dtype=tf.float32)
        self.memory.append((costmap, goal_vel_hed, action, reward, next_costmap, next_goal_vel_hed, done))

    # Initializes the main and target networks, and loads the model weights and last used training parameters if needed
    def initNetwork(self):
        global main_nn
        global target_nn

        rospy.loginfo('Initializing network...')

        # Get a random sample for intialization
        costmap = self.memory[0][0]
        goal_vel_hed = self.memory[0][1]
        action = self.memory[0][2]
        reward = self.memory[0][3]
        next_costmap = self.memory[0][4]
        next_goal_vel_hed = self.memory[0][5]
        done = self.memory[0][6]
    
        main_nn = self.model
        target_nn = self.target_model

        # Run the sample through the network
        train_step(costmap, goal_vel_hed, action, reward, next_costmap, next_goal_vel_hed, done) 

        main_nn = self.target_model
        target_nn = self.model

        # Run the sample through the network
        train_step(costmap, goal_vel_hed, action, reward, next_costmap, next_goal_vel_hed, done)             

        # Load saved weights if needed, either last of previous stage or of a predetermined episode of the current stage
        if self.load_model and self.load_episode > 0:
            rospy.loginfo('Loading saved model...')
            # Load weights of load_episode parameter
            path = self.dirPath + str(self.load_episode) + ".weights"
            with open(path, 'rb') as file:
                weights = pickle.load(file)   
            self.model.set_weights(weights)
            self.target_model.set_weights(weights)

            # Load epsilon of load_episode parameter
            with open(self.dirPath + str(self.load_episode) + '.json') as outfile:
                param = json.load(outfile)
                self.epsilon = param.get('epsilon')

	    self.load_model = False
        elif self.load_model:
            rospy.loginfo('Loading last saved model...')
            # Load last weights
            path = self.lastDirPath + "_last.weights"
            rospy.loginfo(path)
            with open(path, 'rb') as file:
                weights = pickle.load(file)   
            self.model.set_weights(weights)   
            self.target_model.set_weights(weights) 

            # Load last used epsilon
            with open(self.lastDirPath + "_last.json") as outfile:
                param = json.load(outfile)
                self.epsilon = param.get('epsilon')                   
		
	    self.load_model = False

    # Extracts a random batch from the replay buffer and runs the training loop, returns the mean of the losses of this batch
    def trainModel(self):
        global main_nn
        global target_nn
        # Generate a random batch
        mini_batch = random.sample(self.memory, self.batch_size)

        losses = []

        for i in range(self.batch_size):
            # Get next sample
            costmap = mini_batch[i][0]
            goal_vel_hed = mini_batch[i][1]
            action = mini_batch[i][2]
            reward = mini_batch[i][3]
            next_costmap = mini_batch[i][4]
            next_goal_vel_hed = mini_batch[i][5]
            done = mini_batch[i][6]
     
            main_nn = self.model
            target_nn = self.target_model

            # Run forward and back propagation
            loss = train_step(costmap, goal_vel_hed, action, reward, next_costmap, next_goal_vel_hed, done) 
            losses.append(loss)
        
        # Return the mean loss of this batch
        return np.mean(losses)

# The agent that uses pre-trained weights and runs the agent with no exploration
class Agent():
    # Initializes the agent, set action_size to the same action_size that was set in training
    def __init__(self, action_size):
        # Creates a reinforce agent to use some of its functions, load_params set to false as to not load parameters for a full reinforce agent
        self.r_a = ReinforceAgent(action_size, load_params=False)
        # Calculate where is the saved weights
        self.dirPath = os.path.dirname(os.path.realpath(__file__))
        # Load parameters of the stage
        self.stage = rospy.get_param('/stage_number')
        stage_int = int(self.stage)
        # Setting inital pose depending on stage
        self.set_inital_pose(stage_int)
        # Finish weights dir calculation
        self.dirPath = self.dirPath.replace('rl_nav/src', 'rl_nav/save_model/stage_' + str(self.stage) + "_")

        # Build model
        self.model = self.r_a.buildModel()

    def preprocessCostmap(self, costmap, reset_history=False):
        return self.r_a.preprocessCostmap(costmap, reset_history)

    def preprocessGoalAndVelocitiesAndHeading(self, goal_dist, velocities, heading):
        return self.r_a.preprocessGoalAndVelocitiesAndHeading(goal_dist, velocities, heading)        

    # Initializes the model and loads the model weights
    def initNetwork(self, costmap, goal_vel_hed):
        global main_nn
        global target_nn

        rospy.loginfo('Initializing network...')

        main_nn = self.model
        target_nn = self.model

        # Run the sample through the network
        train_step(costmap, goal_vel_hed, 0, 0, costmap, goal_vel_hed, False)            

        rospy.loginfo('Loading last saved model...')
        # Loading last saved model weights of the stage
        path = self.dirPath + "_last.weights"
        rospy.loginfo(path)
        with open(path, 'rb') as file:
            weights = pickle.load(file)   
        self.model.set_weights(weights)   

    # Calculating next action using stage and the pretrained model, no exploration here
    def getAction(self, costmap, goal_vel_hed):      
        self.q_value = self.model(costmap, goal_vel_hed)
        return np.argmax(self.q_value)

    # Setting inital pose of the robot, needed for global planning
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

    # Take m number of local goals from the global plan
    def take(self, array, m):
        local_goals = []
        for i in range(len(array)):
            if i % m == 0:
                local_goals.append(array[i])
        if len(array) - 1 % m != 0:
            local_goals.append(array[-1])
        return local_goals

    # Using the default global planner, create a global plan and take m local goals from it
    def global_plan(self, start_x, start_y, x, y, m):
        # Wait for global planner
        rospy.wait_for_service('/move_base/make_plan')
        # Call global planner
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
        # Return m local goals from the global plan
        return self.take(result.plan.poses, m)         
      
# Main Reinforce agent training node. This function, creates the enviornment and the agent, then it runs 
# the training loop. It also publishes the action and q-value functions for visualization in other nodes.
# For such nodes, see the original turtlebot3 implementation(in the README)
def train():
    # Set up publishers for training visualization
    pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
    pub_get_action = rospy.Publisher('get_action', Float32MultiArray, queue_size=5)
    result = Float32MultiArray()
    get_action = Float32MultiArray()

    # Setting up action size and current loss
    action_size = num_actions
    loss = float('inf')

    # Creating the enviornment, this handles the connection with the actual enviornment(real of gazebo)
    env = Env(action_size)

    # Creating the Reinforce agent
    agent = ReinforceAgent(action_size)
    scores, episodes = [], []
    global_step = 0
    start_time = time.time()

    # The main training loop
    for e in range(agent.load_episode + 1, agent.total_episodes):
        done = False
        # Reseting the enviornment and creating a new goal, lastly getting the current agent state
        costmap, goal_dist, velocities, heading = env.reset()
        costmap = agent.preprocessCostmap(costmap, reset_history=True)
        goal_vel_hed = agent.preprocessGoalAndVelocitiesAndHeading(goal_dist, velocities, heading)
        score = 0
        # The episode training loop
        for t in range(agent.episode_step): 
            # Calculating next action using epsilon greedy           
            action = agent.getAction(costmap, goal_vel_hed)

            # Sending action to the enviornment
            next_costmap, next_goal_dist, next_velocities, next_heading, reward, done = env.step(action)

            # Calculating the actual agent state from the state returned by the enviornment
            next_costmap = agent.preprocessCostmap(next_costmap)
            next_goal_vel_hed = agent.preprocessGoalAndVelocitiesAndHeading(next_goal_dist, next_velocities, next_heading)

            # Appending to the replay buffer
            agent.appendMemory(costmap, goal_vel_hed, action, reward, next_costmap, next_goal_vel_hed, done)

            # Training the model after memory is large enough
            if len(agent.memory) >= agent.train_start:                
                loss = agent.trainModel()                  
                                                  
            # Appending reward to current score, switching between current and next states                                                  
            score += reward
            costmap, goal_vel_hed = next_costmap, next_goal_vel_hed
            get_action.data = [action, score, reward]
            pub_get_action.publish(get_action)

            # Initializes network
            if agent.load_model or (e == 1 and t == 1):
                agent.initNetwork()

                param_keys = ['epsilon']
                param_values = [agent.epsilon]
                param_dictionary = dict(zip(param_keys, param_values))
		
            # Saves weights and current parameters each 10 episodes
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

            # Ends episode after episode_step stesps
            if t >= agent.episode_step - 1:
                rospy.loginfo("Time out!!")
                done = True

            # Logs progress each 10 steps
            if t % 10 == 0:
                m, s = divmod(int(time.time() - start_time), 60)
                h, m = divmod(m, 60)

                rospy.loginfo('Loss: %f, Ep: %d/%d Step: %d score: %.2f memory: %d epsilon: %.2f time: %d:%02d:%02d',
                              loss, e, agent.total_episodes, t, score, len(agent.memory), agent.epsilon, h, m, s)                

            # If episode is done, updates target network, saves current training parameters, logs current progress
            if done:
                rospy.loginfo("DONE! UPDATE TARGET NETWORK")
                result.data = [score, np.max(agent.q_value)]
                pub_result.publish(result)
                agent.updateTargetModel()
                scores.append(score)
                episodes.append(e)
                m, s = divmod(int(time.time() - start_time), 60)
                h, m = divmod(m, 60)

                rospy.loginfo('Loss: %f, Ep: %d/%d score: %.2f memory: %d epsilon: %.2f time: %d:%02d:%02d',
                              loss, e, agent.total_episodes, score, len(agent.memory), agent.epsilon, h, m, s)
                param_keys = ['epsilon']
                param_values = [agent.epsilon]
                param_dictionary = dict(zip(param_keys, param_values))
                break

            global_step += 1
            # if global_step % agent.target_update == 0:
            #     rospy.loginfo("UPDATE TARGET NETWORK")
              

        # Discounting epislon to avoid exploration in the future
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

# Running a trained agent with only local goal planning using the agent
def run():
    # Setting up the agent
    action_size = num_actions
    agent = Agent(action_size)
    # Creating the enviornment
    env = Env(action_size)
    # Reseting the enviornment
    costmap, goal_dist, velocities, heading = env.reset()
    costmap = agent.preprocessCostmap(costmap, reset_history=True)
    goal_vel_hed = agent.preprocessGoalAndVelocitiesAndHeading(goal_dist, velocities, heading)    
    # Initializing model and loading weights
    agent.initNetwork(costmap, goal_vel_hed)
    
    done = False
    rate = rospy.Rate(4)
    rospy.loginfo("Running...")
    # Running local goal planner loop
    while not rospy.is_shutdown():
        # Getting next action with no exploration
        action = agent.getAction(costmap, goal_vel_hed)
        # Acting on the enviornment
        costmap, goal_dist, velocities, heading, _, done = env.step(action)
        costmap = agent.preprocessCostmap(costmap)
        goal_vel_hed = agent.preprocessGoalAndVelocitiesAndHeading(goal_dist, velocities, heading)          

        # If done, due to goal ro colission, start over
        if done:
            rospy.loginfo("DONE!")
            done = False
            costmap, goal_dist, velocities, heading = env.reset()
            costmap = agent.preprocessCostmap(costmap, reset_history=True)
            goal_vel_hed = agent.preprocessGoalAndVelocitiesAndHeading(goal_dist, velocities, heading)                       
        rate.sleep()

# A function to prepare a global plan, using an agent, the enviornment and a modulo to use when deciding how many local goals of the global plan to use
def prepareGlobalPlan(agent, env, plan_modulo_param):
    rospy.loginfo("Preparing global plan...")
    # Getting agent current position
    pos_x, pos_y = env.getPosition()
    # Getting gobal position
    global_goal_x, global_goal_y = env.getGoal()
    rospy.loginfo("\t From: " + str(pos_x) + ", " + str(pos_y))
    rospy.loginfo("\t To: " + str(global_goal_x) + ", " + str(global_goal_y))
    # Creating a global plan
    plan = agent.global_plan(pos_x, pos_y, global_goal_x, global_goal_y, plan_modulo_param)
    rospy.loginfo("-----------------------------------------------------------")
    rospy.loginfo("\t Got plan with: " + str(len(plan)) + " local goals...")
    return plan

# Returns the current distance of the next local goal, seperated to x,y
def get_local_goal_dist(env, local_goal_x, local_goal_y):
    pos_x, pos_y = env.getPosition()

    return local_goal_x - pos_x, local_goal_y - pos_y

# Returns the current distance of the next local goal
def get_local_goal_distance(env, local_goal_x, local_goal_y):
    pos_x, pos_y = env.getPosition()

    goal_distance = round(math.hypot(local_goal_x - pos_x, local_goal_y - pos_y), 2)

    return goal_distance

# Follow a generated global plan
def follow_plan(env, plan, curr_index):
    # If curr_index==-1, this is the start of the plan
    if curr_index == -1:
        rospy.loginfo("Starting to follow plan...")
        curr_index = 0
    # Get next local goal
    next_goal = plan[curr_index]
    # Get next local goal location
    local_goal_x, local_goal_y = next_goal.pose.position.x, next_goal.pose.position.y
    # Calculate next local goal distance
    current_distance = get_local_goal_distance(env, local_goal_x, local_goal_y)
    if curr_index == 0:
        rospy.loginfo("Next local goal is: " + str(local_goal_x) + ", " + str(local_goal_y))
    if current_distance < 0.2:
        # Local goal has been reached, move to the next one
        rospy.loginfo("Local goal reached...")
        curr_index = curr_index + 1
        if curr_index < len(plan):
            next_goal = plan[curr_index]
            local_goal_x, local_goal_y = next_goal.pose.position.x, next_goal.pose.position.y            
            rospy.loginfo("Next local goal is: " + str(local_goal_x) + ", " + str(local_goal_y))
            return curr_index
        else:
            # Global plan has been finished
            rospy.loginfo("Global goal reached...")
            return -1
    return curr_index

# A function to run a trained agent with a global and local reinforce agent planners
def run_with_global_planner():
    # A modulo parameter to determine how many local goals to take from the global plan
    plan_modulo_param = rospy.get_param('/plan_modulo')
    plan_modulo_param = int(plan_modulo_param)
    # Number of actions, should be the same as in training
    action_size = num_actions
    # Creating the agent
    agent = Agent(action_size)
    # Creating the enviornment
    env = Env(action_size)
    # Reseting the enviornment
    costmap, _, velocities, heading = env.reset()    
    # Preparing a global plan
    plan = prepareGlobalPlan(agent, env, plan_modulo_param)
    # Taking the first local goal
    next_goal = plan[0]
    # Extracting x and y of next goal
    local_goal_x, local_goal_y = next_goal.pose.position.x, next_goal.pose.position.y  
    # Calculating next local goal distance      
    goal_dist = get_local_goal_dist(env, local_goal_x, local_goal_y)    
    costmap = agent.preprocessCostmap(costmap, reset_history=True)
    goal_vel_hed = agent.preprocessGoalAndVelocitiesAndHeading(goal_dist, velocities, heading)    
    # Initialzies the agent model
    agent.initNetwork(costmap, goal_vel_hed)
    
    done = False
    rate = rospy.Rate(4)
    curr_index = -1
    rospy.loginfo("Running...")
    # Running the global plan
    while not rospy.is_shutdown():
        # Follow next local goal
        curr_index = follow_plan(env, plan, curr_index)
        # Getting next local goal
        next_goal = plan[curr_index]
        # Extracting x,y of next local goal
        local_goal_x, local_goal_y = next_goal.pose.position.x, next_goal.pose.position.y 
        # Calculating local goal distance       
        goal_dist = get_local_goal_dist(env, local_goal_x, local_goal_y)
        costmap = agent.preprocessCostmap(costmap)
        goal_vel_hed = agent.preprocessGoalAndVelocitiesAndHeading(goal_dist, velocities, heading)          
        # Getting next action
        action = agent.getAction(costmap, goal_vel_hed)
        # Acting on the enviornment
        costmap, _, velocities, heading, _, done = env.step(action)

        # If curr_index==-1 global goal reached
        if curr_index == -1:
            rospy.loginfo("DONE!")
            done = False
            costmap, _, velocities, heading = env.reset()
            costmap = agent.preprocessCostmap(costmap, reset_history=True)
            plan = prepareGlobalPlan(agent, env, plan_modulo_param) 

        rate.sleep()        


# The main function
if __name__ == '__main__':
    # Initializing node    
    rospy.init_node('turtlebot3_dqn')
    # Train or run agent
    train_param = rospy.get_param('/train')
    train_param = int(train_param)
    train_param = train_param == 1
    # Run training
    if train_param:
        rospy.loginfo("Running training...")
        train()
    else:
        # Run local plan agent or global plan agent
        run_type_param = rospy.get_param('/run_type')
        run_type_param = int(run_type_param)
        if run_type_param == 1:
            rospy.loginfo("Running trained agent with local goal...")
            run()
        else:
            rospy.loginfo("Running trained agent with global planner and local goals...")
            run_with_global_planner()            

    
