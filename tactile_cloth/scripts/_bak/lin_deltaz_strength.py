#!/usr/bin/python
import rospy
# from std_msgs.msg import Float64MultiArray, MultiArrayDimension
from std_msgs.msg import String
from sensor_msgs.msg import Joy
# from inputs import get_gamepad
import numpy as np
from serial import Serial
from lin_deltaz_utils import move_to_xyz_pos
import numpy as np
import linear_actuator_pb2
from time import sleep
BTN_Y_INDEX = 3
BTN_B_INDEX = 1
class DeltaZ:
    def __init__(self):
        rospy.init_node('linear_delta_node')
        self.publish_deltaZ = False
        self.vertical_control = False

        self.pub = rospy.Publisher('/record_trial', String, queue_size=10)

        self.trials_to_collect = 0
        # if self.collection_mode == 'joy':
        self.joy_sub = rospy.Subscriber("/joy", Joy, self.joy_callback)
        # elif self.collection_mode == 'auto':
            # self.auto_sub = rospy.Subscriber("/auto_data_collect", String, self.auto_callback)
        # self.joy_data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.arduino = Serial('/dev/ttyACM0', 57600)
        self.delta_message = linear_actuator_pb2.lin_actuator()
        self.delta_message.id = 1
        self.pinch = False
        self.rub = False
        self.is_pinch = rospy.get_param('grasp_type') == 'norub'
        self.is_rub = rospy.get_param('grasp_type') == 'vertrub'
        # self.traj_counter = 0
        self.no_of_rubs=2

    def spin(self):
        count = 0
        
        while not rospy.is_shutdown():            
            if self.pinch == True and self.trials_to_collect > 0:
                self.pub.publish("start")
                sleep(1)

                move_to_xyz_pos([1.7,1,3,-1.7,0,3], self.delta_message, self.arduino)
                sleep(1)
                move_to_xyz_pos([-1.5,1,3,1.5,0,3], self.delta_message, self.arduino)

                sleep(0.1)
                self.pub.publish("stop")
                self.trials_to_collect -= 1
                if self.trials_to_collect == 0:
                    self.pinch = False
                sleep(1)
            if self.rub == True and self.trials_to_collect > 0:                
                time_per = 0.2
                # move_to_xyz_pos([-1.5,1.1,3,1.5,0,3.8], self.delta_message, self.arduino)
                move_to_xyz_pos([0, 0, 0, 0, 0, 0], self.delta_message, self.arduino)
                sleep(0.5)

                self.pub.publish("start")
                sleep(1)
                
                # move_to_xyz_pos([1.5,1.1,3,-1.5,0,3.8], self.delta_message, self.arduino)
                import IPython; IPython.embed()
                move_to_xyz_pos([-1.5, 1.1, 3, -1.5, 0, 3.8], self.delta_message, self.arduino)
                for i in range(self.no_of_rubs):
                    move_to_xyz_pos([1.5,1.1,3,-1.5,0,3.8], self.delta_message, self.arduino)
                    sleep(time_per)
                    move_to_xyz_pos([1.5,1.1,3,-1.5,0,1.8], self.delta_message, self.arduino)
                    sleep(time_per)
                move_to_xyz_pos([-1.5,1,3,1.5,0,3], self.delta_message, self.arduino)

                sleep(0.1)
                self.pub.publish("stop")
                self.trials_to_collect -= 1
                if self.trials_to_collect == 0:
                    self.rub = False
                sleep(1)

    def joy_callback(self,data):
        if(data.buttons[BTN_B_INDEX] == 1):
            self.collection_mode = rospy.get_param('collection_mode')
            self.num_auto_trials = rospy.get_param('num_auto_trials')
            self.trials_to_collect = self.num_auto_trials if self.collection_mode == "auto" else 1
            # self.pinch = True 
            self.pinch = self.is_pinch
            self.rub = self.is_rub
            # self.traj_counter = 0
        # if(data.buttons[BTN_Y_INDEX] == 1):
            # self.rub = True
        # self.joy_data = data.axes[:6]

    # def auto_callback(self, msg):
        # self.rub = True

if (__name__=="__main__"):
    dz = DeltaZ()
    dz.spin()



