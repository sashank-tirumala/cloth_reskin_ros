# Learning to singulate layers of cloth code base

*Disclaimer: This code depends on a specific hardware setup (Franka Control PC to control the Frankas and special hardware for the gripper), thus it will not work out of the box for your hardware setup. This code is best used as a reference.*

This is the official repository to run the robot code for the IROS 2022 Paper : [Learning to Singulate Layers of Cloth using Tactile Feedback](https://sites.google.com/view/reskin-cloth?pli=1). 
The code required to train the Machine Learning Models is present at : 
[IROS2022 Training Code](https://github.com/DanielTakeshi/cloth_reskin)

# Installation
* Install prerequisites: [ROS melodic](http://wiki.ros.org/melodic/Installation/Ubuntu), [catkin_tools](https://catkin-tools.readthedocs.io/en/latest/installing.html)
* Make a catkin workspace `mkdir -p ~/catkin_ws/src && cd ~/catkin_ws && catkin build`
* Clone repo into catkin src `git clone  git@github.com:sashank-tirumala/LSLC_robot_code.git`
* Clone `franka_ros` submodule: `git submodule update --init`
* Build repo `catkin_make`
## Running the code
   
To run specific commands look at the readme files inside ```singlearm_ros```, ```tactile_cloth``` folders

## Citation

If you find the code helpful, consider citing the following paper:

```
@inproceedings{tirumala2022,
  title     = {{Learning to Singulate Layers of Cloth based on Tactile Feedback}},
  author    = {Sashank Tirumala and Thomas Weng and Daniel Seita and Oliver Kroemer and Zeynep Temel and David Held},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year      = {2022},
}
```