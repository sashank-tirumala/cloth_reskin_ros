# Learning to singulate layers of cloth code base

This is the official repository to run the robot code for the IROS 2022 Paper : [Learning to Singulate Layers of Cloth using Tactile Feedback](https://sites.google.com/view/reskin-cloth?pli=1). 

# Installation
* Install prerequisites: [ROS melodic](http://wiki.ros.org/melodic/Installation/Ubuntu), [catkin_tools](https://catkin-tools.readthedocs.io/en/latest/installing.html)
* Make a catkin workspace `mkdir -p ~/catkin_ws/src && cd ~/catkin_ws && catkin build`
* Clone repo into catkin src `git clone  git@github.com:sashank-tirumala/LSLC_robot_code.git`
* Clone `franka_ros` submodule: `git submodule update --init`
* Build repo `catkin_make`
## Setting up the code
   
To run specific commands look at the readme files inside ```singlearm_ros```, ```tactile_cloth``` folders