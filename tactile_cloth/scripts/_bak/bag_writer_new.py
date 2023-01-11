#!/usr/bin/python
import os
import time
import rospy
from std_msgs.msg import String
import subprocess
import signal
import yaml

class RosbagWriter:
    def __init__(self):
        rospy.init_node('rosbag_writer')
        self.rec_exp_sub = rospy.Subscriber('/record_experiment', String, self.record_experiment_callback)
        self.rec_trial_sub = rospy.Subscriber('/record_trial', String, self.record_trial_callback)
        self.rec_suc__sub = rospy.Subscriber('/record_success', String, self.success_callback)
        self.shutdown = rospy.on_shutdown(self.process_shutdown)
        self.proc = None
        self.postprocess_children = []

    def create_exp_folder()
    def create_trial_bag(self, data):
        _, _, _, i = data.split('_')
        self.save_dir = rospy.get_param('save_dir')
        self.cloth_type = rospy.get_param('cloth_type')
        self.grasp_type = rospy.get_param('grasp_type')
        self.collection_mode = rospy.get_param('collection_mode')
        self.exp_name = rospy.get_param('exp_name')
        self.random_grasp_pos = rospy.get_param('random_grasp_pos')
        grasp_name = self.grasp_type = "random" if self.random_grasp_pos else self.grasp_type
        self.trial_type = '%s_%s_%s_%s' % (self.cloth_type, grasp_name, self.collection_mode, i)

        # Create dir for all trials of this type if missing
        self.trial_type_dir = "%s/%s/%s" % (self.save_dir, self.exp_name, self.trial_type)
        if not os.path.exists(self.trial_type_dir):
            rospy.logwarn("trial type dir does not exist: " + self.trial_type_dir)
            rospy.logwarn("Attempting to create")
            os.makedirs(self.trial_type_dir)        
            rospy.logwarn("Created")

        # Make directory for this particular run, timestamped
        self.trial_name = "%s_%s" % (
            time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime()),
            self.trial_type
        )
        os.mkdir("%s/%s" % (self.trial_type_dir, self.trial_name))

        # Init bag
        self.bag_name = '%s/%s/data.bag' % (
            self.trial_type_dir, self.trial_name
        )

    def save_cfg(self, data):
        command, height_diff, slide_diff, i = data.split('_')
        data_dict = {
            'command': command,
            'height_diff': float(height_diff),
            'slide_diff': float(slide_diff),
            'i': int(i)
        }
        with open('%s/%s/config.yaml' % (self.trial_type_dir, self.trial_name), 'w') as yaml_file:
            yaml.dump(data_dict, yaml_file, default_flow_style=False)

    def success_callback(self, msg):
        with open('%s/%s/success_%s.txt' % (self.trial_type_dir, self.trial_name, msg.data), 'w') as txt_file:
            txt_file.write(msg.data)

    def record_callback(self, msg):
        # rospy.loginfo(msg.data)
        if msg.data.startswith('start'):
            rospy.loginfo("Recording trial")
            self.create_trial_bag(msg.data.replace('start_', ''))
            self.save_cfg(msg.data.replace('start_', ''))
            if self.proc is not None:
                rospy.logerr("Error: ovewriting proc var")
                import IPython; IPython.embed()
            self.proc = subprocess.Popen(["rosbag", "record", "/reskin", "/webcam_image", "/grasp_startstop", "/classifier_commands", "/classifier_output",
                "-O", self.bag_name], preexec_fn=os.setsid)
        elif msg.data == 'stop':
            if self.proc is not None:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                # if ret != 0:
                    # rospy.logerr("Error: rosbag process not killed")
                    # import IPython; IPython.embed()
                rospy.loginfo("Recording stopped")
                rospy.loginfo("Processing bag")
                trial_path = '%s/%s' % (self.trial_type_dir, self.trial_name)
                dir_path = os.path.dirname(os.path.abspath(__file__))
                postprocess_child = subprocess.Popen([
                    "python2.7", 
                    "%s/process_bag.py" % dir_path, 
                    "--path", trial_path, 
                    "--predict",
                    ], 
                    preexec_fn=os.setsid)
                self.postprocess_children.append(postprocess_child)
        else:
            raise NotImplementedError

    def process_shutdown(self):
        if self.proc is not None:
            os.killpg(self.proc.pid, signal.SIGINT)

        for child in self.postprocess_children:
            poll = child.poll()
            if poll is None:
                os.killpg(child.pid, signal.SIGINT)

if __name__ == '__main__':
    rw = RosbagWriter()
    rospy.spin()