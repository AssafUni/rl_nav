# Robot Navigation with Map-Based Deep Reinforcement Learning
https://arxiv.org/pdf/2002.04349.pdf

An implementation for the paper with a modified reward function and reduced number of actions.
Forked from 
https://github.com/ROBOTIS-GIT/turtlebot3_machine_learning.git
Added Dobule Dueling DQN with the aid of 
https://markelsanz14.medium.com/introduction-to-reinforcement-learning-part-4-double-dqn-and-dueling-dqn-b349c9a61ea1



As Part of a course by Ronen Brafman(https://www.cs.bgu.ac.il/~brafman/) from Ben Gurion University.

## Installation

Follow TurtleBot3 Manual for kinetic:
https://emanual.robotis.com/docs/en/platform/turtlebot3/quick-start/
And the machine learning section for kinetic:
https://emanual.robotis.com/docs/en/platform/turtlebot3/machine_learning/#machine-learning
Run:
```sh
pip install tensorflow==2.1.*
```
## How to Run 
Open up a shell, go to the catkin workspace. Copy project to workspace. Run:
```sh
git clone -b kinetic-devel https://github.com/ROBOTIS-GIT/turtlebot3_simulations.git
catkin_make
source devel/setup.bash
export TURTLEBOT3_MODEL=burger
roslaunch rl_nav turtlebot3_stage_1.launch
```
Open another shell and Run:
```sh
export TURTLEBOT3_MODEL=burger
roslaunch rl_nav navigation_123.launch
```
Open another shell and Run:
```sh
export TURTLEBOT3_MODEL=burger
chmod +x src/turtlebot3_dqn.py
roslaunch rl_nav turtlebot3_dqn_stage_1.launch
```
Once done, run next stage(end all shells first):

Shell #1:
```sh
export TURTLEBOT3_MODEL=burger
roslaunch rl_nav turtlebot3_stage_2.launch
```
Shell #2:
```sh
export TURTLEBOT3_MODEL=burger
roslaunch rl_nav navigation_123.launch
```
Shell #3:
```sh
export TURTLEBOT3_MODEL=burger
roslaunch rl_nav turtlebot3_dqn_stage_2.launch
```

Once done, run next stage(end all shells first):

Shell #1:
```sh
export TURTLEBOT3_MODEL=burger
roslaunch rl_nav turtlebot3_stage_3.launch
```
Shell #2:
```sh
export TURTLEBOT3_MODEL=burger
roslaunch rl_nav navigation_123.launch
```
Shell #3:
```sh
export TURTLEBOT3_MODEL=burger
roslaunch rl_nav turtlebot3_dqn_stage_3.launch
```

Once done, run next stage(end all shells first):

Shell #1:
```sh
export TURTLEBOT3_MODEL=burger
roslaunch rl_nav turtlebot3_stage_4.launch
```
Shell #2:
```sh
export TURTLEBOT3_MODEL=burger
roslaunch rl_nav navigation_4.launch
```
Shell #3:
```sh
export TURTLEBOT3_MODEL=burger
roslaunch rl_nav turtlebot3_dqn_stage_4.launch
```

Once done, we can run a trained agent with a global planner:

Shell #1:
```sh
export TURTLEBOT3_MODEL=burger
roslaunch rl_nav turtlebot3_stage_4.launch
```
Shell #2:
```sh
export TURTLEBOT3_MODEL=burger
roslaunch rl_nav navigation_4.launch
```
Shell #3:
```sh
export TURTLEBOT3_MODEL=burger
turtlebot3_dqn_stage_4_run_global.launch
```

## Project structure and file explanation

1. src directory-
    1. turtlebot3_dqn.py- The main node of the Double Dueling DQN. It runs the enviornment, the reinforce agent and the training loop.
    2. environment.py- A code responsible to represent the enviornment. The DQN node creates the enviornment and handles the interaction between the agent and the enviornment. In essence, the enviornment.py is the piece of code responsible to interact with the actual gazebo/real enviornment.
    3. respawnGoal.py- The code responsible for instantiating the goal and deleting it.
    4. *-obstacle.py- The obstacle code files manage the complicated moving obstacles in stages 3 and 4.
2. launch directory-
    1. turtlebot3_stage_*.launch- Launch files for each stage of the training. 
        1. gui- Use gui:=true to open a gazebo gui, false to do otherwise.
    2. navigation_123.launch- The Launch file for the navigation nodes, use for stages 1 to 3. This lanuch file will also load the prepard map file for the stages.
        1. open_rviz- Use open_rviz:=true to open a gui of rviz, false to do otherwise.
    3. navigation_4.launch- The Launch file for the navigation nodes, use for stage 4. This lanuch file will also load the prepard map file for the stage.
        1. open_rviz- Use open_rviz:=true to open a gui of rviz, false to do otherwise.
    4. turtlebot3_dqn_stage_*.launch- The launch file for the DQN node, enviornment and training. Each stage if set to load the last saved weights of the last stage. Otherwise, use the parameters of the launch file.
        1. load_model- Use load_model:=true to load a model, this will be automatically true for stages 2 to 4.
        2. load_episode- Use load_episode:=## (number of episode) to load weights of specific episode. If parameter load_model is set to true manaully, the loaded weights are of the same stage, otherwise the loaded weights are of the previous stage. If load_episode is set to zero, the last weights of the stage are loaded.
    5. turtlebot3_dqn_stage_*_run.launch- The launch file to run a trained agent. This will run a local goal planner only using the last weights of the stage.
    6. turtlebot3_dqn_stage_*_run_global.launch- The launch file to run a trained agent with global planner. This will run a local goal planner using the last weights of the stage, but it will use a global planner to create local goal points instead of goal directly to the goal.
3. map directory-
    1. map123- The map for stages 1 to 3.
    2. map4- The map for stage 4.
4. models directory- This directory includes some models for the enviornment.
5. save_model directory- This is where the weights of the model will be saved for each 10 episodes.
