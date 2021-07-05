# Robot Navigation with Map-Based Deep Reinforcement Learning
https://arxiv.org/pdf/2002.04349.pdf

An implementation for the paper.
Forked from 
https://github.com/ROBOTIS-GIT/turtlebot3_machine_learning.git
Added Dobule Dueling DQN with the aid of 
https://markelsanz14.medium.com/introduction-to-reinforcement-learning-part-4-double-dqn-and-dueling-dqn-b349c9a61ea1

As Part of a course by Ronen Brafman(https://www.cs.bgu.ac.il/~brafman/) from Ben Gurion University.

TODO:
1. Test and debug all stages on a GPU(too slow on CPU).
2. Add a global planner and integrate it with local planner.

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
catkin_make
source devel/setup.bash
export TURTLEBOT3_MODEL=burger
roslaunch rl_nav turtlebot3_stage_1.launch
```
Open another shell and Run:
```sh
export TURTLEBOT3_MODEL=burger
roslaunch turtlebot3_navigation turtlebot3_navigation.launch
```
Open another shell and Run:
```sh
export TURTLEBOT3_MODEL=burger
roslaunch rl_nav turtlebot3_dqn_stage_1.launch
```
Once done, run next stage(end all shells first):
Shell #1:
```sh
export TURTLEBOT3_MODEL=burger
roslaunch rl_nav turtlebot3_stage_1.launch
```
Shell #2:
```sh
export TURTLEBOT3_MODEL=burger
roslaunch turtlebot3_navigation turtlebot3_navigation.launch
```
Shell #3:
```sh
export TURTLEBOT3_MODEL=burger
roslaunch rl_nav turtlebot3_dqn_stage_2.launch
```
And so fourth.

## License

MIT

**Free Software, Hell Yeah!**

