#! /usr/bin/env python
import rospy
import moveit_commander
from geometry_msgs.msg import PoseStamped


def initScene(scene, frame_id="panda_link0"):
    scene.remove_world_object("table")
    scene.remove_world_object("wall")

    time_stamp = rospy.get_rostime()
    table_size = [1, 1.5, 0.01]
    table_pose = PoseStamped()
    table_pose.header.frame_id = frame_id
    table_pose.header.stamp = time_stamp
    table_pose.pose.orientation.w = 1.0
    table_pose.pose.position.x = 0.6
    table_pose.pose.position.y = 0.3
    table_pose.pose.position.z = 0.005
    rospy.sleep(0.5)

    wall_size = [0.1, 1, 1]
    wall_pose = PoseStamped()
    wall_pose.header.frame_id = frame_id
    wall_pose.header.stamp = time_stamp
    wall_pose.pose.orientation.w = 1.0
    wall_pose.pose.position.x = -0.5
    wall_pose.pose.position.y = -0.3
    wall_pose.pose.position.z = 0.5
    rospy.sleep(0.5)

    scene.add_box("table", table_pose, table_size)
    scene.add_box("wall", wall_pose, wall_size)


rospy.init_node("scene_geometry")
scene = moveit_commander.PlanningSceneInterface(ns="")
initScene(scene)
