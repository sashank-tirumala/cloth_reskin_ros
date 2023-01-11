Single Arm
===

# Calibrating the wrist camera

# Setting collision boundaries

# Running 

Start roscore on primary pc
* `roscore`

On the control pc (dopey)
* `roslaunch panda_moveit_config panda_control_moveit_dual.launch robot_ip:=172.16.0.2 robot_id:=panda_1 launch_rviz:=false`

On the workhorse pc (yertle)
* `roslaunch singlearm_ros sensor_rviz.launch`
* `roslaunch singlearm_ros pick_place.launch`

Send reset action
* rostopic pub reset std_msgs/String ''

rostopic pub /reset/goal bimanual_ros/ResetActionGoal "header:
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