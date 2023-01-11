# Debugging Commands
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
