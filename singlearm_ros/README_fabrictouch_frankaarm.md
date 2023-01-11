Fabric Touch README
===

# Installation
* Install prerequisites: [ROS melodic](http://wiki.ros.org/melodic/Installation/Ubuntu), [catkin_tools](https://catkin-tools.readthedocs.io/en/latest/installing.html)
* Make a catkin workspace `mkdir -p ~/catkin_ws/src && cd ~/catkin_ws && catkin build`
* Clone repo into catkin src `git clone git@github.com:thomasweng15/bimanual_folding.git -b fabric_touch`
* Clone `franka_ros` submodule: `git submodule update --init`
* Build repo `catkin build`

# Setup
* Set collision geometries 
  * Start RViz with motion planning using instructions in Execution section (up to step #3)
  * Set collision geometries in `init_collision_geometries.py` script  
    * Alternatively, set collision geometries in RViz under 'Scene Objects', save to file in RViz, and load file
    * If collision geometries do not update in RViz, turn the MotionPlanning plugin on/off to update
* Set initial joint angle pose
  * Start RViz and motion planning using instructions in Execution section (up to step #3)
  * Move arm using gravity compensation mode to desired pose
  * Get joint angles from `rostopic echo joint_states` and save to `config/config.yaml`
* Attach the delta gripper and reskin
  * Update URDF and collision geometries to account for delta gripper (see /config for .urdf and .xacro files)
  * See tactile_cloth repo for instructions on delta + reskin setup
* Check args of launch files to make sure data is being saved to the right place

# Execution
1. Start roscore on primary pc `roscore`
2. On franka control pc
  * `source ~/ws1/devel/setup.bash`
  * `roslaunch panda_moveit_config panda_control_moveit_rviz.launch robot_ip:=172.16.0.2 launch_rviz:=false load_gripper:=false`
3. Launch roslaunch files on primary `roslaunch singlearm_ros grasp_policy.launch`. WARNING: robot will move to reset position, make sure collision geometries of the robot and the environment are set!
5. Execute grasp by publishing to rostopic `rostopic pub -1 /move_and_grasp std_msgs/String "run"`
6. Reset arm position by publishing to rostopic: `rostopic pub -1 /reset std_msgs/String "run"`

# Debugging 
Debug move_fingers action
```
rostopic pub -1 /move_fingers/goal singlearm_ros/MoveFingersActionGoal "header: 
  seq: 0
  stamp:
    secs: 0
    nsecs: 0
  frame_id: ''
goal_id:
  stamp:
    secs: 0
    nsecs: 0
  id: ''
goal:
  command:
    data: 'vertrub'"
```

Debug move_action_server grasp action
```
rostopic pub -1 /grasp/goal singlearm_ros/GraspActionGoal "header:
  seq: 0
  stamp:
    secs: 0
    nsecs: 0
  frame_id: ''
goal_id:
  stamp:
    secs: 0
    nsecs: 0
  id: '0'"
```

Debug move_action_server reset action 
```
rostopic pub -1 /reset/goal singlearm_ros/ResetActionGoal "header:
  seq: 0
  stamp:
    secs: 0
    nsecs: 0
  frame_id: ''
goal_id:
  stamp:
    secs: 0
    nsecs: 0
  id: '0'"
```
