#!/usr/bin/env python
import os
import cv2
import rospy
import actionlib
import message_filters
import numpy as np
import constants
np.set_printoptions(suppress=True)

from copy import deepcopy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float32
from geometry_msgs.msg import Pose, Quaternion

# from utils.marker_visualizer import MarkerVisualizer
from singlearm_ros.msg import MoveAndGraspAction, MoveAndGraspGoal, ResetAction, ResetGoal
# FoldDualAction, FoldDualGoal

import time
import tf
from tf.transformations import *

# import torch
# import torchvision
# import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from std_msgs.msg import Int64

import argparse

class GraspServer(object):
    def __init__(self):
        rospy.init_node('grasp_server')

        rospy.loginfo("Waiting for reset action server...")
        self.reset_client = actionlib.SimpleActionClient('reset', ResetAction)
        self.reset_client.wait_for_server()
        rospy.loginfo("Connected to reset action server.")

        rospy.loginfo("Waiting for grasp action server...")
        self.moveandgrasp_client = actionlib.SimpleActionClient('move_and_grasp', MoveAndGraspAction)
        self.moveandgrasp_client.wait_for_server()
        rospy.loginfo("Connected to grasp action server.")

        self.sub1 = rospy.Subscriber('run_experiment', String, self.run_experiment)
        self.sub2 = rospy.Subscriber('reset', String, self.reset)

        self.clf_preds = []
        self.clf_sub = rospy.Subscriber("/classifier_output", Int64, self.clf_callback)

        self.success_pub = rospy.Publisher('success_trial', String, queue_size=1)
        self.record_pub = rospy.Publisher('record_trial', String, queue_size=1)
        self.dbg_img_pub = rospy.Publisher('dbg_image_record', String, queue_size=1)

    def clf_callback(self, msg):
        self.clf_preds.append(msg.data)

    def reset(self, ask_response=True):
        goal = ResetGoal()
        resp = "1" if ask_response else "0"
        goal.ask_response = String(resp)
        self.reset_client.send_goal(goal)
        self.reset_client.wait_for_result()
        response = self.reset_client.get_result()
        rospy.loginfo(response)

    def manual_grasp(self, msg):
        """
        Expects msg to be command_heightdiff_slidediff 
        """
        self.create_exp_dir()        
        self.dbg_img_pub.publish("start_dbg_record")

        command, height_diff, slide_diff = msg.data.split('_')
        height_diff = float(height_diff)
        slide_diff = float(slide_diff)
        rospy.loginfo("Received grasp request with %f height diff and %f slide_diff" % (height_diff, slide_diff))
        self.record_pub.publish("start_%s_%f_%f_0" % (command, height_diff, slide_diff))
        self.move_and_grasp(command=command, height_diff=height_diff, slide_diff=slide_diff)
        self.record_pub.publish("stop")

        self.dbg_img_pub.publish("end_dbg_record")

    def create_exp_dir(self):
        self.save_dir = rospy.get_param('/app/save_dir')
        self.cloth_type = rospy.get_param('/app/cloth_type')
        self.grasp_type = rospy.get_param('/app/grasp_type')
        self.collection_mode = rospy.get_param('/app/collection_mode')
        self.exp_name = rospy.get_param('/app/exp_name')
        self.random_grasp_pos = rospy.get_param('/app/random_grasp_pos')
        grasp_name = self.grasp_type = "random" if self.random_grasp_pos else self.grasp_type
        self.trial_type = '%s_%s_%s' % (self.cloth_type, grasp_name, self.collection_mode)

        if not os.path.exists("%s/%s" % (self.save_dir, self.exp_name)):
            os.makedirs("%s/%s" % (self.save_dir, self.exp_name))

        runs = os.listdir("%s/%s" % (self.save_dir, self.exp_name))
        samelayer_runs = [x for x in runs if self.cloth_type in x]
        run_idx = len(samelayer_runs)

        # Create dir for all trials of this type if missing
        self.trial_type_dir = "%s/%s/%s_exp%d" % (self.save_dir, self.exp_name, self.trial_type, run_idx)
        os.makedirs(self.trial_type_dir)
        rospy.set_param("trial_type_dir", self.trial_type_dir)

    def baseline_grasp(self, msg, random=False):
        """
        Expects msg to be numlayers to grasp
        """
        self.create_exp_dir()
        self.dbg_img_pub.publish("start_dbg_record")

        rospy.logwarn("Warning: running baseline grasp policy")
        goal_layers = int(msg.data)
        success = False
        for i in range(self.num_attempts):
            self.clf_preds = []
            grasp_command = rospy.get_param("/robot_app/grasp_command")
            command = "move+%s" % grasp_command

            # Choose next action
            if random:
                height_absolute = np.random.uniform(low=0.135, high=0.165)
                rospy.loginfo("randomly selected height: %f" % height_absolute)
            else:
                if goal_layers == 1:
                    height_absolute = .1575 + np.random.uniform(low=-0.005, high=0.005)
                elif goal_layers == 2:
                    height_absolute = .14 + np.random.uniform(low=-0.005, high=0.005)
                rospy.loginfo("open loop height with noise: %f" % height_absolute)
            slide_diff = -0.04 # fixed slide diff, could be random?

            self.record_pub.publish("start_%s_%f_%f_%d" % (command, height_absolute, slide_diff, i))
            self.move_and_grasp(command=command, height_diff=None, slide_diff=slide_diff, height_absolute=height_absolute)

            val, counts = np.unique(self.clf_preds, return_counts=True)
            pred = val[0]
            if float(counts[0]) / len(self.clf_preds) < 0.5:
                rospy.logwarn("Classifier output is noisy")
                pass # continue 
            elif pred == goal_layers:
                success = True
                break

            # move back if not successful 
            self.move_and_grasp(command="unpinch+move_out", height_diff=0, slide_diff=-slide_diff)
            self.record_pub.publish("stop")
            time.sleep(1)
        
        if success:
            self.success_pub.publish("1")
            self.move_and_grasp(command="move", height_diff=-0.05, slide_diff=0.0)
            self.record_pub.publish("stop")
            self.dbg_img_pub.publish("end_dbg_record")
            self.reset(ask_response=False)
        else:
            self.success_pub.publish("0")
            self.dbg_img_pub.publish("end_dbg_record")

    def closedloop_grasp(self, msg):
        """
        Expects msg to be numlayers to grasp
        """
        self.create_exp_dir()    
        self.dbg_img_pub.publish("start_dbg_record")

        rospy.logwarn("Warning: running closedloop grasp policy")
        goal_layers = int(msg.data)

        grasp_command = rospy.get_param("/robot_app/grasp_command")

        # initial action
        height_diff = 0.01
        slide_diff = -0.04 #Increased sliding

        # move to top of stack
        # height_absolute = .167 # + np.random.uniform(low=-0.005, high=0.005) # horizontal
        height_absolute = .265 # + np.random.uniform(low=-0.005, high=0.005)
        self.move_and_grasp(command="move", height_diff=None, slide_diff=0.0, height_absolute=height_absolute)

        success = False
        for i in range(self.num_attempts):
            self.clf_preds = []
            command = "move+%s" % grasp_command
            # command = "move+vertrub%s" % grasp_command
            self.record_pub.publish("start_%s_%f_%f_%d" % (command, height_diff, slide_diff, i))
            self.move_and_grasp(command=command, height_diff=height_diff, slide_diff=slide_diff)
            
            # self.move_and_grasp(command='unpinch', height_diff=None, slide_diff=None)
            # self.move_and_grasp(command='move', height_diff=0.01, slide_diff=0.0)

            if grasp_command == 'pinchmoverub':
                self.clf_preds = []
                self.move_and_grasp(command='pinch', height_diff=None, slide_diff=None)
            prev_height_diff = height_diff
            prev_slide_diff = slide_diff

            # check result, compute next action
            if(constants.RANDOM_HEIGHT_FLAG):
                increment = constants.INCREMENT + np.random.normal(loc=0.0, scale=0.0001) #Adding randomness to height movement (Does it improve or not?). Reduced to 0.003
            else:
                increment = constants.INCREMENT

            val, counts = np.unique(self.clf_preds, return_counts=True)
            # pred = val[0]
            pred = -1 # Hard code
            if float(counts[0]) / len(self.clf_preds) < 0.5:
                rospy.logwarn("Classifier output is noisy")
                # if np.random.randint(2) == 0: # Increment in random direction
                #     height_diff += increment
                # else:
                #     height_diff -= increment
            # elif pred == goal_layers:
            if pred == goal_layers:
                success = True
                break
            else: 
                if pred > goal_layers: # move up
                    height_diff -= increment
                    rospy.loginfo("move up")
                elif pred < goal_layers: # move down
                    height_diff += increment
                    rospy.loginfo("move down")
        
            # move back if not successful 
            self.move_and_grasp(command="unpinch+move_out", height_diff=-prev_height_diff, slide_diff=-prev_slide_diff)
            self.record_pub.publish("stop")
            time.sleep(1)

        if success:
            self.success_pub.publish("1")
            self.move_and_grasp(command="move", height_diff=-0.05, slide_diff=0.0)
            self.record_pub.publish("stop") #Why did we comment this out before
            self.dbg_img_pub.publish("end_dbg_record")
            self.reset(ask_response=False)
        else:
            self.success_pub.publish("0")
            self.dbg_img_pub.publish("end_dbg_record")

    def run_experiment(self, msg):
        self.num_attempts = rospy.get_param('/robot_app/num_exp_attempts') # number of attempts per trial

        policy = rospy.get_param('/robot_app/policy')
        if policy == 'manual':
            self.manual_grasp(msg)
        elif policy == 'random':
            self.baseline_grasp(msg, random=True)
        elif policy == 'openloop':
            self.baseline_grasp(msg, random=False)
        elif policy == 'closedloop':
            self.closedloop_grasp(msg)

    def move_and_grasp(self, command, height_diff, slide_diff, height_absolute=None):
        height_diff_f32 = Float32(height_diff) if height_diff is not None else Float32(-100.) # no support for NaN
        height_absolute_f32 = Float32(height_absolute) if height_absolute is not None else Float32(-100.)
        goal = MoveAndGraspGoal()
        goal.command = String(command)
        goal.height_diff = height_diff_f32
        goal.slide_diff = Float32(slide_diff)
        goal.height_absolute = height_absolute_f32
        self.moveandgrasp_client.send_goal(goal)
        result = self.moveandgrasp_client.wait_for_result()
        return self.moveandgrasp_client.get_state()
        
if __name__ == '__main__':
    s = GraspServer()
    rospy.spin()
