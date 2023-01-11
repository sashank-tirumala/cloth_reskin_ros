#!/usr/bin/env python
import os
import rospy
import actionlib
import numpy as np

np.set_printoptions(suppress=True)

from std_msgs.msg import String, Float32

from singlearm_pkg.msg import (
    MoveAndGraspAction,
    MoveAndGraspGoal,
    ResetAction,
    ResetGoal,
)

from tf.transformations import *


class GraspServer(object):
    def __init__(self):
        rospy.init_node("grasp_server")

        rospy.loginfo("Waiting for reset action server...")
        self.reset_client = actionlib.SimpleActionClient("reset", ResetAction)
        self.reset_client.wait_for_server()
        rospy.loginfo("Connected to reset action server.")

        rospy.loginfo("Waiting for grasp action server...")
        self.moveandgrasp_client = actionlib.SimpleActionClient(
            "move_and_grasp", MoveAndGraspAction
        )
        self.moveandgrasp_client.wait_for_server()
        rospy.loginfo("Connected to grasp action server.")

        self.data_sub = rospy.Subscriber("collect_data", String, self.data_collect)
        self.sub2 = rospy.Subscriber("reset", String, self.reset)

    def data_collect(self, msg):
        self.create_exp_dir()
        self.x_dist = float(rospy.get_param("/robot_app/x_dist"))
        self.x_dist = 0.025
        self.num_data_points = 9
        self.slide_dist = 0.005
        if msg.data == "start":
            increment = self.x_dist / self.num_data_points
            for j in range(5):
                for i in range(int(self.num_data_points)):
                    print(i)
                    goal = MoveAndGraspGoal()
                    goal.command = String("datacollect_pinch")
                    goal.height_diff = Float32(0.0)
                    goal.slide_diff = Float32(0.0)
                    goal.x_diff = Float32(np.random.uniform(-0.02, 0.02))
                    self.moveandgrasp_client.send_goal(goal)
                    self.moveandgrasp_client.wait_for_result()
                    goal.command = String("move_x")
                    goal.height_diff = Float32(0.0)
                    goal.slide_diff = Float32(0.0)
                    goal.x_diff = Float32(increment)
                    self.moveandgrasp_client.send_goal(goal)
                    self.moveandgrasp_client.wait_for_result()
                goal = MoveAndGraspGoal()
                goal.command = String("move")
                goal.height_diff = Float32(0.0)
                goal.slide_diff = Float32(self.slide_dist)
                goal.x_diff = Float32(0.0)
                self.moveandgrasp_client.send_goal(goal)
                self.moveandgrasp_client.wait_for_result()
                increment = -1 * increment

    def reset(self, msg):
        goal = ResetGoal()
        self.reset_client.send_goal(goal)
        self.reset_client.wait_for_result()
        response = self.reset_client.get_result()
        rospy.loginfo(response)

    def create_exp_dir(self):
        self.save_dir = rospy.get_param("/app/save_dir")
        self.cloth_type = rospy.get_param("/app/cloth_type")
        self.grasp_type = rospy.get_param("/app/grasp_type")
        self.collection_mode = rospy.get_param("/app/collection_mode")
        self.exp_name = rospy.get_param("/app/exp_name")
        self.random_grasp_pos = rospy.get_param("/app/random_grasp_pos")
        grasp_name = self.grasp_type = (
            "random" if self.random_grasp_pos else self.grasp_type
        )
        self.trial_type = "%s_%s_%s" % (
            self.cloth_type,
            grasp_name,
            self.collection_mode,
        )

        if not os.path.exists("%s/%s" % (self.save_dir, self.exp_name)):
            os.makedirs("%s/%s" % (self.save_dir, self.exp_name))

        # Create dir for all trials of this type if missing
        self.trial_type_dir = "%s/%s/%s" % (
            self.save_dir,
            self.exp_name,
            self.trial_type,
        )
        if not os.path.exists("%s" % (self.trial_type_dir)):
            os.makedirs(self.trial_type_dir)
        rospy.set_param("trial_type_dir", self.trial_type_dir)


if __name__ == "__main__":
    s = GraspServer()
    rospy.spin()
