#!/usr/bin/env python
import os
import rospy
import actionlib
import numpy as np
import constants

np.set_printoptions(suppress=True)

from std_msgs.msg import String, Float32, Int64

from singlearm_pkg.msg import (
    MoveAndGraspAction,
    MoveAndGraspGoal,
    ResetAction,
    ResetGoal,
)

import time
from tf.transformations import *


import yaml
from video import VideoRecorder


class GraspServer(object):
    def __init__(self):
        rospy.init_node("grasp_server")

        self.reset_client = actionlib.SimpleActionClient("reset", ResetAction)
        self.reset_client.wait_for_server()
        self.moveandgrasp_client = actionlib.SimpleActionClient(
            "move_and_grasp", MoveAndGraspAction
        )
        self.moveandgrasp_client.wait_for_server()

        self.sub1 = rospy.Subscriber("run_experiment", String, self.run_experiment)

        self.clf_preds = []
        self.img_clf_preds = []
        self.clf_sub = rospy.Subscriber("/classifier_output", Int64, self.clf_callback)
        self.im_clf_sub = rospy.Subscriber(
            "/img_classifier_output", Int64, self.img_clf_callback
        )

        self.record_pub = rospy.Publisher("record_trial", String, queue_size=1)
        self.dbg_img_pub = rospy.Publisher("dbg_image_record", String, queue_size=1)

        self.is_random_exp = rospy.get_param("/robot_app/run_type") == "run_random_exp"
        if self.is_random_exp:
            self.init_rand_exp()

        self.use_dslr = rospy.get_param("use_dslr")
        if self.use_dslr:
            self.vr = VideoRecorder()
            rospy.on_shutdown(self.shutdown)

        time.sleep(1)
        self.reset(ask_response=True)

    def shutdown(self):
        self.vr.stop_movie()

    def clf_callback(self, msg):
        self.clf_preds.append(msg.data)

    def img_clf_callback(self, msg):
        self.img_clf_preds.append(msg.data)

    def reset_clf_preds(self):
        self.clf_preds = []
        self.img_clf_preds = []

    def reset(self, ask_response=True):
        goal = ResetGoal()
        resp = "1" if ask_response else "0"
        goal.ask_response = String(resp)
        self.reset_client.send_goal(goal)
        self.reset_client.wait_for_result()
        response = self.reset_client.get_result()

    def move_and_grasp(self, command, height_diff, slide_diff, height_absolute=None):
        height_diff_f32 = (
            Float32(height_diff) if height_diff is not None else Float32(-100.0)
        )  # no support for NaN
        height_absolute_f32 = (
            Float32(height_absolute) if height_absolute is not None else Float32(-100.0)
        )
        goal = MoveAndGraspGoal()
        goal.command = String(command)
        goal.height_diff = height_diff_f32
        goal.slide_diff = Float32(slide_diff)
        goal.height_absolute = height_absolute_f32
        self.moveandgrasp_client.send_goal(goal)
        result = self.moveandgrasp_client.wait_for_result()
        return self.moveandgrasp_client.get_state()

    def init_rand_exp(self):
        """
        Create dir for random experiment
        save_dir / timestamp + exp_name / methods / trials
        Save results for all methods in this dir
        """
        self.save_dir = rospy.get_param("/app/save_dir")

        # If exp dir is provided, use it, else create new
        exp_name = rospy.get_param("/app/exp_name")
        self.rand_exp_folder = rospy.get_param("/exp_cfg/rand_exp_dir")
        if self.rand_exp_folder == "":
            ts = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            self.rand_exp_folder = ts + "_" + exp_name
            os.makedirs("%s/%s" % (self.save_dir, self.rand_exp_folder))

            # Save config
            names = rospy.get_param_names()
            params = [x for x in names if "/exp_cfg" in x]
            data_dict = {}
            for key in params:
                data_dict[key] = rospy.get_param(key)

            with open(
                "%s/%s/exp_cfg.yaml" % (self.save_dir, self.rand_exp_folder), "w"
            ) as yaml_file:
                yaml.dump(data_dict, yaml_file, default_flow_style=False)
        else:
            # Load config from dir
            rospy.logwarn("%s/%s/exp_cfg.yaml" % (self.save_dir, self.rand_exp_folder))
            with open(
                "%s/%s/exp_cfg.yaml" % (self.save_dir, self.rand_exp_folder), "r"
            ) as yaml_file:
                data_dict = yaml.safe_load(yaml_file)
            for key in data_dict.keys():
                rospy.set_param(key, data_dict[key])
        self.rand_exp_path = "%s/%s" % (self.save_dir, self.rand_exp_folder)
        rospy.loginfo("Saving into %s" % self.rand_exp_path)

        # Get current counts of methods
        self.num_trials_per_method = rospy.get_param("exp_cfg/num_trials_per_method")
        self.methods = rospy.get_param("exp_cfg/methods")
        self.method_names = self.methods.keys()
        for name in self.method_names:
            method_dir = "%s/%s" % (self.rand_exp_path, name)
            if not os.path.exists(method_dir):
                os.makedirs(method_dir)

    def run_experiment(self, msg):
        # Check if experiment is done
        methods_to_sample = []
        cloth_type = rospy.get_param("/app/cloth_type")
        self.num_trials_per_method = rospy.get_param("exp_cfg/num_trials_per_method")
        for name in self.method_names:
            method_dir = "%s/%s/%s" % (self.save_dir, self.rand_exp_folder, name)
            num_trials = len(os.listdir(method_dir))

            rospy.logwarn("num trials: %s %d " % (method_dir, num_trials))

            if num_trials < self.num_trials_per_method:
                methods_to_sample.append(name)
            elif num_trials > self.num_trials_per_method:
                rospy.logerr(
                    "More than %d trials for method %s"
                    % (self.num_trials_per_method, name)
                )

        rospy.logwarn(methods_to_sample)
        if methods_to_sample == []:
            rospy.loginfo("Experiment complete")
            rospy.signal_shutdown()
            return

        # Randomly select method and set rosparams
        for method_name in np.random.permutation(methods_to_sample):
            if "image" in method_name and not rospy.get_param(
                "/robot_app/imgclf_ready"
            ):
                continue
            else:
                break

        rospy.loginfo("Method %s trial" % method_name)
        for key, value in self.methods[method_name]["rosparams"]:
            rospy.set_param(key, value)

        # Run method
        self.create_trial_dir(method_name)
        policy = rospy.get_param("/robot_app/policy")
        goal_layers = int(rospy.get_param("/app/cloth_type").replace("cloth", ""))
        if policy == "random":
            self.baseline_grasp(goal_layers, random=True)
        elif policy == "openloop":
            self.baseline_grasp(goal_layers, random=False)
        elif policy == "closedloop":
            self.closedloop_grasp(goal_layers)

    def create_trial_dir(self, name):
        ts = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        cloth_type = rospy.get_param("/app/cloth_type")
        trial_dirname = "%s_%s" % (ts, cloth_type)
        self.trial_dir = "%s/%s/%s" % (self.rand_exp_path, name, trial_dirname)
        os.makedirs(self.trial_dir)
        rospy.set_param("trial_type_dir", self.trial_dir)

    def baseline_grasp(self, goal_layers, random=False):
        """
        Expects msg to be numlayers to grasp
        """
        if self.use_dslr:
            self.vr.start_movie(save_dir=self.trial_dir)
        self.dbg_img_pub.publish("start_dbg_record")
        rospy.logwarn("Warning: running baseline grasp policy")
        success = False
        num_attempts = rospy.get_param(
            "/robot_app/num_exp_attempts"
        )  # number of attempts per trial
        clf_type = rospy.get_param("/app/classifier_type")
        clf_pred_success = False
        for i in range(num_attempts):
            self.reset_clf_preds()
            grasp_command = rospy.get_param("/robot_app/grasp_command")
            command = "move+%s" % grasp_command

            # Choose next action
            if random:
                randrange = rospy.get_param("/robot_app/random")
                height_absolute = np.random.uniform(low=randrange[0], high=randrange[1])
                rospy.loginfo("randomly selected height: %f" % height_absolute)
            else:
                openloop = rospy.get_param("robot_app/openloop")
                ol_noise = rospy.get_param("robot_app/openloop_noise")
                if goal_layers == 1:
                    if ol_noise != 0:
                        rospy.logwarn("Using noisy open loop")
                    height_absolute = openloop[0] + np.random.uniform(
                        low=-ol_noise, high=+ol_noise
                    )
                elif goal_layers == 2:
                    if ol_noise != 0:
                        rospy.logwarn("Using noisy open loop")
                    height_absolute = openloop[1] + np.random.uniform(
                        low=-ol_noise, high=+ol_noise
                    )
                elif goal_layers == 3:  # mainly for data collection
                    if ol_noise != 0:
                        rospy.logwarn("Using noisy open loop")
                    height_absolute = openloop[2] + np.random.uniform(
                        low=-ol_noise, high=+ol_noise
                    )

                rospy.loginfo("open loop height: %f" % height_absolute)
            slide_diff = -0.04

            self.record_pub.publish(
                "start_%s_%f_%f_%d" % (command, height_absolute, slide_diff, i)
            )
            self.move_and_grasp(
                command=command,
                height_diff=None,
                slide_diff=slide_diff,
                height_absolute=height_absolute,
            )

            if grasp_command == "pinchmoverub" or grasp_command == "robustifypinch":
                self.reset_clf_preds()
                self.move_and_grasp(command="close", height_diff=None, slide_diff=None)

            if not random:  # openloop baseline, runs once and terminates
                clf_pred_success = True
                break

            if clf_type == "tactile":
                rospy.logwarn("Using tactile classifier")
                clf_preds = self.clf_preds
            else:
                rospy.logwarn("Using image classifier")
                clf_preds = self.img_clf_preds
            val, counts = np.unique(clf_preds, return_counts=True)
            index = np.argmax(counts)
            pred = val[index]
            rospy.logwarn(
                "pred: %d goal: %d num preds: %d" % (pred, goal_layers, len(clf_preds))
            )
            rospy.logwarn(clf_preds)
            if float(counts[index]) / len(clf_preds) < 0.5:
                rospy.logwarn("Classifier output is noisy")

            if pred == goal_layers:
                clf_pred_success = True
                break

            # move back if not successful
            self.move_and_grasp(
                command="unpinch+move_out", height_diff=0, slide_diff=-slide_diff
            )
            self.record_pub.publish("stop")
            time.sleep(0.5)

        if clf_pred_success:
            self.move_and_grasp(command="movedown", height_diff=-0.05, slide_diff=0.0)
            self.record_pub.publish("stop")
            self.dbg_img_pub.publish("end_dbg_record")
            if self.use_dslr:
                self.vr.stop_movie()
            self.reset(ask_response=False)
        else:
            self.dbg_img_pub.publish("end_dbg_record")
            if self.use_dslr:
                self.vr.stop_movie()

    def closedloop_grasp(self, goal_layers):
        """
        Expects msg to be numlayers to grasp
        """
        if self.use_dslr:
            self.vr.start_movie(save_dir=self.trial_dir)
        self.dbg_img_pub.publish("start_dbg_record")

        rospy.logwarn("Warning: running closedloop grasp policy")

        grasp_command = rospy.get_param("/robot_app/grasp_command")

        # initial action
        height_diff = 0.0
        slide_diff = -0.04

        # move to top of stack
        closedloop = rospy.get_param("robot_app/closedloop")
        height_absolute = closedloop[0] if goal_layers == 1 else closedloop[1]
        self.move_and_grasp(
            command="move",
            height_diff=None,
            slide_diff=0.0,
            height_absolute=height_absolute,
        )

        success = False
        num_attempts = rospy.get_param(
            "/robot_app/num_exp_attempts"
        )  # number of attempts per trial
        clf_type = rospy.get_param("/app/classifier_type")
        clf_pred_success = False
        for i in range(num_attempts):
            self.reset_clf_preds()
            command = "move+%s" % grasp_command
            self.record_pub.publish(
                "start_%s_%f_%f_%d" % (command, height_diff, slide_diff, i)
            )
            self.move_and_grasp(
                command=command, height_diff=height_diff, slide_diff=slide_diff
            )

            if grasp_command == "pinchmoverub" or grasp_command == "robustifypinch":
                self.reset_clf_preds()
                self.move_and_grasp(command="close", height_diff=None, slide_diff=None)

            prev_height_diff = height_diff
            prev_slide_diff = slide_diff

            # check result, compute next action
            increment = rospy.get_param("/robot_app/increment")
            if constants.RANDOM_HEIGHT_FLAG:
                increment += np.random.normal(
                    loc=0.0, scale=0.0001
                )  # Adding randomness to height movement (Does it improve or not?). Reduced to 0.003

            if clf_type == "tactile":
                rospy.logwarn("Using tactile classifier")
                clf_preds = self.clf_preds
            else:
                rospy.logwarn("Using image classifier")
                clf_preds = self.img_clf_preds
            val, counts = np.unique(clf_preds, return_counts=True)
            index = np.argmax(counts)
            pred = val[index]
            if float(counts[index]) / len(clf_preds) < 0.5:
                rospy.logwarn("Classifier output is noisy")

            rospy.logwarn(
                "pred: %d goal: %d num preds: %d" % (pred, goal_layers, len(clf_preds))
            )
            if pred == goal_layers:
                clf_pred_success = True
                break
            else:
                if pred > goal_layers:  # move up
                    height_diff -= increment
                    rospy.loginfo("move up")
                elif pred < goal_layers:  # move down
                    height_diff += increment
                    rospy.loginfo("move down")

            # move back if not successful
            self.move_and_grasp(
                command="unpinch+move_out",
                height_diff=-prev_height_diff,
                slide_diff=-prev_slide_diff,
            )
            self.record_pub.publish("stop")
            time.sleep(1)

        if clf_pred_success:
            self.move_and_grasp(command="move", height_diff=-0.05, slide_diff=0.0)
            self.record_pub.publish("stop")
            self.dbg_img_pub.publish("end_dbg_record")
            if self.use_dslr:
                self.vr.stop_movie()
            self.reset(ask_response=False)
        else:
            self.dbg_img_pub.publish("end_dbg_record")
            if self.use_dslr:
                self.vr.stop_movie()


if __name__ == "__main__":
    s = GraspServer()
    rospy.spin()
