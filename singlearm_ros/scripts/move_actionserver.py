#!/usr/bin/env python
import rospy
import actionlib
import moveit_commander
import moveit_msgs
from copy import deepcopy
from std_msgs.msg import String
from singlearm_ros.msg import ResetAction, ResetActionResult, MoveAndGraspAction, MoveAndGraspActionResult, MoveFingersAction, MoveFingersGoal
import tf
import numpy as np
from geometry_msgs.msg import Pose
from actionlib_msgs.msg import GoalStatus
import tf.transformations as tfs
import constants
import rosparam

def get_waypoints(curr_pose, goal_pose):
    waypoints = []
    diff_x = goal_pose.position.x - curr_pose.position.x
    diff_y = goal_pose.position.y - curr_pose.position.y
    diff_z = goal_pose.position.z - curr_pose.position.z
    for i in range(5):
        curr_pose.position.x += diff_x / 5.0
        curr_pose.position.y += diff_y / 5.0
        curr_pose.position.z += diff_z / 5.0
        waypoints.append(deepcopy(curr_pose))
    return waypoints

def retime_jt(plan, scale):
    for i in range(len(plan.joint_trajectory.points)):
        velocities = []
        accelerations = []
        for j in range(len(plan.joint_trajectory.points[i].positions)):
            velocities.append(plan.joint_trajectory.points[i].velocities[j] * scale) 
            accelerations.append(plan.joint_trajectory.points[i].accelerations[j] * scale*scale)
        plan.joint_trajectory.points[i].velocities = velocities
        plan.joint_trajectory.points[i].accelerations = accelerations
        duration = plan.joint_trajectory.points[i].time_from_start / scale
        plan.joint_trajectory.points[i].time_from_start = duration
    return plan

class MoveActionServer:
    def __init__(self):
        rospy.init_node("move_actionserver")

        self.max_vel_scale_factor = 0.2
        self.max_acc_scale_factor = 0.2
        self.planning_time = 2.0

        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface(ns='')
        self.group = moveit_commander.MoveGroupCommander('panda_arm', robot_description='robot_description', ns='')
        self.group.set_max_velocity_scaling_factor(self.max_vel_scale_factor)
        self.group.set_max_acceleration_scaling_factor(self.max_acc_scale_factor)
        self.group.set_planner_id("RRTstarkConfigDefault")
        self.group.set_planning_time(self.planning_time)

        self.movefingers_client = actionlib.SimpleActionClient('move_fingers', MoveFingersAction)
        self.movefingers_client.wait_for_server()

        self.reset_server = actionlib.SimpleActionServer('reset', ResetAction, execute_cb=self.reset, auto_start=False)
        self.reset_server.start()

        self.moveandgrasp_server = actionlib.SimpleActionServer('move_and_grasp', MoveAndGraspAction, execute_cb=self.move_action, auto_start=False)
        self.moveandgrasp_server.start()
        self.min_depth = 0.187 # Old_value
        self.min_depth = 0.108 #New_height
        self.init_pose = None

    def reset(self, goal):
        rospy.logwarn("received reset request")
        ask_response = goal.ask_response.data
        result = ResetActionResult()
        pose = Pose()
        if(constants.IS_HORIZONTAL == False):
            q = tfs.quaternion_from_euler(-4*np.pi/6, np.pi/4, 0, 'rxzy')
            pose.position.x = 0.6
            pose.position.y = 0.0
            if(constants.RESET_RANDOM):
                pose.position.z = np.random.uniform(low=0.19, high=0.21)
            else:
                pose.position.z = 0.20
        else:
            # q = tfs.quaternion_from_euler(np.radians(-100), np.radians(45), 0, 'rxzy')
            q = tfs.quaternion_from_euler(np.radians(-120), np.radians(45), 0, 'rxzy')
            pose.position.x = 0.6
            pose.position.y = 0.035
            if(constants.RESET_RANDOM):
                # pose.position.z = np.random.uniform(low=0.155, high=0.165)
                pose.position.z = np.random.uniform(low=0.23, high=0.25)
                rospy.logwarn("reset pose %f" % pose.position.z)
            else:
                # pose.position.z = 0.16
                pose.position.z = 0.25
        pose.orientation.x = q[0]
        pose.orientation.y = q[1]
        pose.orientation.z = q[2]
        pose.orientation.w = q[3]
        
        joints = [-1.2449559813509092, 1.7519530168667172, 1.5313610843353174, -1.9609845540565354, -0.8552755286296206, 2.1569541281179867, 1.229067765260736]
        plan = self.group.plan(joints)
        # plan = self.group.plan(pose)
        # if ask_response == "1":
        #     resp = raw_input("Execute? [y/n]")
        #     if resp != 'y':
        #         rospy.logwarn("Execution cancelled")
        #         self.reset_server.set_aborted(result)
        #         return
        
        if not self.move_fingers('open'):
            self.reset_server.set_aborted(result)
            return
        self.group.execute(plan, wait=True)
        self.group.stop()
        self.reset_server.set_succeeded(result)

    def move_fingers(self, command):
        goal = MoveFingersGoal()
        goal.command = String(data=command)
        self.movefingers_client.send_goal(goal)
        self.movefingers_client.wait_for_result()
        response = self.movefingers_client.get_state()
        if response != GoalStatus.SUCCEEDED:
            rospy.logerr('Moving gripper failed.')
            return False
        return True

    def update_height(self, init_pose, height_diff):
        if height_diff == 0:
            return init_pose, []
    
        pose = deepcopy(init_pose)
        if pose.position.z - height_diff < self.min_depth: # account for min valid depth
            clipped_height_diff = pose.position.z - self.min_depth
        else:
            clipped_height_diff = height_diff
        rospy.loginfo("clipped height diff: %f, %f" % (clipped_height_diff, self.min_depth))
        pose.position.z -= clipped_height_diff
        return pose, get_waypoints(init_pose, pose)
        
    def update_slide(self, init_pose, slide_diff):
        if slide_diff == 0:
            return init_pose, []

        pose = deepcopy(init_pose)
        pose.position.y -= slide_diff
        return pose, get_waypoints(init_pose, pose)
    
    def update_trajectory(self, init_pose, post_action=False):
        motion_key = rospy.get_param('/robot_app/motion_key')
        
        paramlist = rosparam.load_file('/home/tweng/tactile_ws/src/bimanual_folding/singlearm_ros/config/config.yaml')
        motion_name = paramlist[0][0]['robot_app']['motions'][motion_key]
        for params, ns in paramlist:
            rosparam.upload_params(ns, params)
        
        motion_type = motion_name["type"]
        all_waypoints = []
        action_type = "actions" if not post_action else "post_actions"
        if motion_type == 'discrete':
            list_of_actions = motion_name[action_type]
            pose = deepcopy(init_pose)
            for action in list_of_actions:
                pose.position.x -= action['diff_x'] 
                pose.position.y -= action['diff_y']
                pose.position.z -= action['diff_z']
                waypoints = get_waypoints(init_pose, pose)
                all_waypoints = all_waypoints + waypoints
        
        return pose, all_waypoints

    def update_x(self, init_pose, x_diff):
        if x_diff == 0:
            return init_pose, []

        pose = deepcopy(init_pose)
        pose.position.x += x_diff
        return pose, get_waypoints(init_pose, pose)

    def move(self, height_diff, ask_response=False, height_absolute=None, scoop=False, slide_diff=None, saveinit=True):
        curr_pose = self.group.get_current_pose().pose     
        if saveinit:
            self.init_pose = deepcopy(curr_pose)
        if height_diff == -100. and height_absolute != -100.: # no support for NaN
            height_diff = curr_pose.position.z - height_absolute

        pose1, waypoints1 = self.update_height(curr_pose, height_diff)
        if scoop:
            pose2, waypoints2 = self.update_trajectory(pose1)
        else:
            pose2, waypoints2 =  self.update_slide(pose1, slide_diff)

        waypoints = []
        if waypoints1 != []:
            waypoints += waypoints1
        if waypoints2 != []:
            waypoints += waypoints2
        if waypoints == []:
            rospy.logwarn("No change for move out")
            return False

        (plan, fraction) = self.group.compute_cartesian_path(
                                waypoints,   # waypoints to follow
                                0.01,        # eef_step
                                5.00)        # jump_threshold
        if fraction != 1.0:
            rospy.logwarn("Path infeasible, fraction planned: {}".format(fraction))
            return False
        
        plan = retime_jt(plan, self.max_vel_scale_factor)

        if(ask_response):
            resp = raw_input("Execute? [y/n]")
            if resp != 'y':
                rospy.logwarn("Execution cancelled")
                return False

        if not self.group.execute(plan, wait=True):
            return False
        return True

    def move_after(self):
        curr_pose = self.group.get_current_pose().pose     

        _, waypoints = self.update_trajectory(curr_pose, post_action=True)
        if waypoints == []:
            rospy.logwarn("No change for move out")
            return False

        (plan, fraction) = self.group.compute_cartesian_path(
                                waypoints,   # waypoints to follow
                                0.01,        # eef_step
                                5.00)        # jump_threshold
        if fraction != 1.0:
            rospy.logwarn("Path infeasible, fraction planned: {}".format(fraction))
            return False
        
        plan = retime_jt(plan, self.max_vel_scale_factor*2)

        if not self.group.execute(plan, wait=True):
            return False
        return True

    def move_out(self):
        curr_pose = self.group.get_current_pose().pose     
        assert self.init_pose is not None

        # get amount to slide out (y)
        pose = deepcopy(curr_pose)
        pose.position.y = self.init_pose.position.y
        pose.position.z = self.init_pose.position.z
        waypoints = get_waypoints(curr_pose, pose)
        
        if waypoints == []:
            rospy.logwarn("No change for move out")
            return False

        (plan, fraction) = self.group.compute_cartesian_path(
                                waypoints,   # waypoints to follow
                                0.01,        # eef_step
                                5.00)        # jump_threshold
        if fraction != 1.0:
            rospy.logwarn("Path infeasible, fraction planned: {}".format(fraction))
            return False
        
        plan = retime_jt(plan, self.max_vel_scale_factor)

        # if ask_response:
        #     resp = raw_input("Execute? [y/n]")
        #     if resp != 'y':
        #         rospy.logwarn("Execution cancelled")
        #         return False

        if not self.group.execute(plan, wait=True):
            return False
        return True
    
    def move_x(self, x_diff, ask_response=False):
        "Negative is towards the robot body, total length 0.013"
        curr_pose = self.group.get_current_pose().pose
        future_pose, waypoints = self.update_x(curr_pose, x_diff)
        (plan, fraction) = self.group.compute_cartesian_path(
                                waypoints,   # waypoints to follow
                                0.01,        # eef_step
                                5.00)        # jump_threshold
        if fraction != 1.0:
            rospy.logwarn("Path infeasible, fraction planned: {}".format(fraction))
            return False
        
        plan = retime_jt(plan, self.max_vel_scale_factor)

        if(ask_response):
            resp = raw_input("Execute? [y/n]")
            if resp != 'y':
                rospy.logwarn("Execution cancelled")
                return False

        if not self.group.execute(plan, wait=True):
            return False
        return True

    def move_action(self, goal):
        """
        Execute a full cycle
            grasp: open fingers, move down, slide, close fingers
            ungrasp: open fingers, slide backwards 
        """
        move_result = MoveAndGraspActionResult()
        command = goal.command.data # [move, move+pinch, unpinch+move]
        height_diff = goal.height_diff.data
        slide_diff = goal.slide_diff.data
        height_absolute = goal.height_absolute.data # for random grasp policy
        x_diff = goal.x_diff.data
        rospy.loginfo('Received %s request with grasp params height diff %s' % (command, height_diff))
        if height_diff != -100. and abs(height_diff) > 0.3:
            rospy.logwarn("policy parameters out of safe range, aborting execution")
            self.moveandgrasp_server.set_aborted(move_result)
            return

        if command == 'move':
            if not self.move(height_diff, height_absolute=height_absolute, slide_diff=slide_diff):
                self.moveandgrasp_server.set_aborted(move_result)
                return
        elif command == 'movedown':
            if not self.move(height_diff, height_absolute=height_absolute, slide_diff=slide_diff, saveinit=False):
                self.moveandgrasp_server.set_aborted(move_result)
                return
        elif command == 'pinch':
            if not self.move_fingers('open'):
                self.moveandgrasp_server.set_aborted(move_result)
                return
            if not self.move_fingers('close'):
                self.moveandgrasp_server.set_aborted(move_result)
                return
        elif command == 'close':
            if not self.move_fingers('close'):
                self.moveandgrasp_server.set_aborted(move_result)
                return
        elif command == 'robustify':
            if not self.move_fingers('robustify'):
                self.moveandgrasp_server.set_aborted(move_result)
                return
        elif command == 'move+fingersforward':
            if not self.move_fingers('open'):
                self.moveandgrasp_server.set_aborted(move_result)
                return
            if not self.move(height_diff, scoop=True):
                self.moveandgrasp_server.set_aborted(move_result)
                return
            if not self.move_fingers('close'):
                self.moveandgrasp_server.set_aborted(move_result)
                return
            if not self.move_fingers('moveforward'):
                self.moveandgrasp_server.set_aborted(move_result)
                return
        elif command == 'move+pinch':
            if not self.move_fingers('open'):
                self.moveandgrasp_server.set_aborted(move_result)
                return
            if not self.move(height_diff, height_absolute=height_absolute, scoop=True):
                self.moveandgrasp_server.set_aborted(move_result)
                return
            if not self.move_fingers('close'):
                self.moveandgrasp_server.set_aborted(move_result)
                return
        elif command == 'move+robustifypinch':
            if not self.move_fingers('open'):
                self.moveandgrasp_server.set_aborted(move_result)
                return
            if not self.move(height_diff, height_absolute=height_absolute, scoop=True):
                self.moveandgrasp_server.set_aborted(move_result)
                return
            if not self.move_fingers('robustify'):
                self.moveandgrasp_server.set_aborted(move_result)
                return
            # if not self.move_fingers('close'):
            #     self.moveandgrasp_server.set_aborted(move_result)
            #     return
        elif command == 'move+pinchmoverub':
            if not self.move_fingers('open'):
                self.moveandgrasp_server.set_aborted(move_result)
                return
            if not self.move(height_diff, height_absolute=height_absolute, scoop=True):
                self.moveandgrasp_server.set_aborted(move_result)
                return
            if not self.move_fingers('close'):
                self.moveandgrasp_server.set_aborted(move_result)
                return
            if not self.move_fingers('open'):
                self.moveandgrasp_server.set_aborted(move_result)
                return
            if not self.move_after():
                self.moveandgrasp_server.set_aborted(move_result)
                return
            if not self.move_fingers('vertrub'):
                self.moveandgrasp_server.set_aborted(move_result)
                return
        elif command == 'move+vertrubmove':
            if not self.move_fingers('open'):
                self.moveandgrasp_server.set_aborted(move_result)
                return
            if not self.move(height_diff, height_absolute=height_absolute, scoop=True):
                self.moveandgrasp_server.set_aborted(move_result)
                return
            if not self.move_fingers('vertrub'):
                self.moveandgrasp_server.set_aborted(move_result)
                return
            if not self.move_after():
                self.moveandgrasp_server.set_aborted(move_result)
                return
        elif command == 'move+vertrubpinch':
            if not self.move_fingers('open'):
                self.moveandgrasp_server.set_aborted(move_result)
                return
            if not self.move(height_diff, height_absolute=height_absolute, scoop=True):
                self.moveandgrasp_server.set_aborted(move_result)
                return
            if not self.move_fingers('vertrub'):
                self.moveandgrasp_server.set_aborted(move_result)
                return
            if not self.move_fingers('close'):
                self.moveandgrasp_server.set_aborted(move_result)
                return
        elif command == 'move+horzrubpinch':
            raise NotImplementedError
        # elif command == 'unpinch+move':
        #     if not self.move_fingers('open'):
        #         self.moveandgrasp_server.set_aborted(move_result)
        #         return

        #     if not self.move(height_diff, height_first=False):
        #     # if not self.move(height_diff, slide_diff, height_first=False):
        #         self.moveandgrasp_server.set_aborted(move_result)
        #         return
        elif command == 'unpinch+move_out':
            if not self.move_fingers('open'):
                self.moveandgrasp_server.set_aborted(move_result)
                return

            if not self.move_out():
                self.moveandgrasp_server.set_aborted(move_result)
                return
        elif command == 'reset':
            self.reset()
        
        elif command == 'move_x':
            if not self.move_x(x_diff):
                self.moveandgrasp_server.set_aborted(move_result)
                return
        elif command == 'move_x+pinch':
            if not self.move_x(x_diff):
                self.moveandgrasp_server.set_aborted(move_result)
                return
            if not self.move_fingers('pinch'):
                self.moveandgrasp_server.set_aborted(move_result)
                return
        elif command == 'datacollect_pinch':
            if not self.move_fingers('pinch'):
                self.moveandgrasp_server.set_aborted(move_result)
                return
        # elif command == 'slide':
        #     if not self.move_from_config("scooping1"):
        #         self.moveandgrasp_server.set_aborted(move_result)
        #         return
        #     if not self.move_fingers('close'):
        #         self.moveandgrasp_server.set_aborted(move_result)
        #         return
        
        else:
            raise NotImplementedError

        self.moveandgrasp_server.set_succeeded(move_result)

if __name__ == '__main__':
    r = MoveActionServer()
    rospy.spin()

        # # close gripper
        # if 'pinch' in command:
        #     response = self.move_fingers('close')
        #     if response != GoalStatus.SUCCEEDED:
        #         rospy.logerr('Closing gripper failed.')
        #         self.moveandgrasp_server.set_aborted(move_result)
        #         self.record_pub.publish("stop")
        #         return

        # if command != 'pinchlift': # open gripper
        #     response = self.move_fingers('open')
        #     if response != GoalStatus.SUCCEEDED:
        #         rospy.logerr('Closing gripper failed.')
        #         self.moveandgrasp_server.set_aborted(move_result)
        #         self.record_pub.publish("stop")
        #         return
        # else: # lift if success, release if fail
        #     # Currently only release is implemented
        #     curr_pose = self.group.get_current_pose().pose
        #     pose1 = deepcopy(curr_pose)
        #     pose1.position.y += slide_diff
        #     pose2 = deepcopy(pose1)
        #     pose2.position.z += clipped_height_diff
        #     if clipped_height_diff == 0:    
        #         waypoints = get_waypoints(curr_pose, pose2)
        #     else:
        #         waypoints1 = get_waypoints(curr_pose, pose1)
        #         waypoints2 = get_waypoints(pose1, pose2)
        #         waypoints = waypoints1 + waypoints2

        #     (plan, fraction) = self.group.compute_cartesian_path(
        #                             waypoints,   # waypoints to follow
        #                             0.01,        # eef_step
        #                             5.00)        # jump_threshold
        #     if fraction != 1.0:
        #         rospy.logwarn("Path infeasible, fraction planned: {}".format(fraction))
        #         self.moveandgrasp_server.set_aborted(move_result)
        #         self.record_pub.publish("stop")
        #         return

        #     plan = retime_jt(plan, self.max_vel_scale_factor)

        #     # resp = raw_input("Execute? [y/n]")
        #     # if resp != 'y':
        #     #     rospy.logwarn("Execution cancelled")
        #     #     self.moveandgrasp_server.set_aborted(move_result)
        #     #     self.record_pub.publish("stop")
        #     #     return
            
        #     # open gripper
        #     response = self.move_fingers('open')
        #     if response != GoalStatus.SUCCEEDED:
        #         rospy.logerr('Opening gripper failed.')
        #         self.moveandgrasp_server.set_aborted(move_result)
        #         self.record_pub.publish("stop")
        #         return

        #     if not self.group.execute(plan, wait=True):
        #         self.moveandgrasp_server.set_aborted(move_result)
        #         self.record_pub.publish("stop")
        #         return