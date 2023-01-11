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
        self.rec_sub = rospy.Subscriber('/record_trial', String, self.record_callback)
        self.success_sub = rospy.Subscriber('/success_trial', String, self.success_callback)
        self.shutdown = rospy.on_shutdown(self.process_shutdown)
        self.proc = None
        self.postprocess_children = []

    def create_trial_bag(self, data):
        _, _, _, i = data.split('_')
        self.trial_type_dir = rospy.get_param("trial_type_dir")
        rospy.logwarn(self.trial_type_dir)

        if not os.path.exists(self.trial_type_dir):
            rospy.logwarn("trial type dir does not exist: " + self.trial_type_dir)
            rospy.logwarn("Attempting to create")
            os.makedirs(self.trial_type_dir)      
            rospy.logwarn("Created")

        # Make directory for this particular run, timestamped
        self.trial_name = "%s_%s" % (
            time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime()),
            "attempt%s" % i
        )
        os.mkdir("%s/%s" % (self.trial_type_dir, self.trial_name))

        # Init bag
        self.bag_name = '%s/%s/data.bag' % (
            self.trial_type_dir, self.trial_name
        )

    def save_cfg(self, data):
        command, height_diff, slide_diff, i = data.split('_')
        data_dict = {
            'command_cfg': {
                'command': command,
                'height_diff': float(height_diff),
                'slide_diff': float(slide_diff),
                'i': int(i)        
            }
        }

        # Save relevant rosparams
        names = rospy.get_param_names()
        params = [x for x in names if '/app' in x or '/robot_app' in x or '/exp_cfg' in x] 
        for key in params:
            data_dict[key] = rospy.get_param(key)

        with open('%s/%s/config.yaml' % (self.trial_type_dir, self.trial_name), 'w') as yaml_file:
            yaml.dump(data_dict, yaml_file, default_flow_style=False)

    def success_callback(self, msg):
        with open('%s/%s/success_%s.txt' % (self.trial_type_dir, self.trial_name, msg.data), 'w') as txt_file:
            txt_file.write(msg.data)

    def record_callback(self, msg):
        if msg.data.startswith('start'):
            rospy.loginfo("Recording trial")
            self.create_trial_bag(msg.data.replace('start_', ''))
            self.save_cfg(msg.data.replace('start_', ''))
            if self.proc is not None:
                rospy.logerr("Error: ovewriting proc var, killing old proc, should probably exit")
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                time.sleep(0.2)
            self.proc = subprocess.Popen(["rosbag", "record", "/reskin", "/webcam_image", "/grasp_startstop", "/classifier_commands", "/classifier_output",
                "-O", self.bag_name], preexec_fn=os.setsid)
        elif msg.data == 'stop':
            if self.proc is not None:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                # time.sleep(5.0)
                # os.killpg(self.proc.pid, signal.SIGINT)
                self.proc = None
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