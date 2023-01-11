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
from singlearm_ros.msg import MoveFingersAction, MoveFingersActionResult
import actionlib

BTN_Y_INDEX = 3
BTN_B_INDEX = 1

class DeltaZ:
    def __init__(self):
        rospy.init_node('linear_delta_node')
        self.joy_sub = rospy.Subscriber("/joy", Joy, self.joy_callback)

        self.arduino = Serial('/dev/ttyACM1', 57600)
        self.delta_message = linear_actuator_pb2.lin_actuator()
        self.delta_message.id = 1
        self.state = {'open':False, 'close':False, 'move_right':False, 'move_left':False, 'move_up':False, 'move_down':False, 'reduce_pinch':False, 'increase_pinch':False}
        self.no_of_rubs = 2

    
    def spin():
        pinch_val = 1.7
        y_offset = 0
        z_offset = 3
        sleep_amt = 0.1
        base_pos = np.array([1.7,1,3,-1.7,0,3])
        while not rospy.is_shutdown():
            if(self.state['open']):
                base_pos[0] = pinch_val
                base_pos[3] = -pinch_val
                move_to_xyz_pos(base_pos, self.delta_message, self.arduino)
                sleep(sleep_amt)
                self.state['open'] = False

            if(self.state['close']):
                base_pos[0] = -pinch_val
                base_pos[3] = pinch_val
                move_to_xyz_pos(base_pos, self.delta_message, self.arduino)
                sleep(sleep_amt)
                self.state['close'] = False

            if(self.state['move_left']):
                base_pos[1] = base_pos[1]+0.05
                move_to_xyz_pos(base_pos, self.delta_message, self.arduino)
                sleep(sleep_amt)
                self.state['move_left'] = False

            if(self.state['move_right']):
                base_pos[1] = base_pos[1]-0.05
                move_to_xyz_pos(base_pos, self.delta_message, self.arduino)
                sleep(sleep_amt)
                self.state['move_right'] = False

            if(self.state['move_up']):
                base_pos[1] = base_pos[2]+0.05
                move_to_xyz_pos(base_pos, self.delta_message, self.arduino)
                sleep(sleep_amt)
                self.state['move_up'] = False

            if(self.state['move_down']):
                base_pos[1] = base_pos[2]-0.05
                move_to_xyz_pos(base_pos, self.delta_message, self.arduino)
                sleep(sleep_amt)
                self.state['move_down'] = False

            if(self.state['reduce_pinch']):
                pinch_val = pinch_val - 0.05
                self.state['reduce_pinch'] = False
                
            if(self.state['increase_pinch']):
                pinch_val = pinch_val + 0.05
                self.state['increase_pinch'] = False
            
            rospy.loginfo("base_pos:",base_pos)

    def joy_callback(self, data):
        print("inside joy callback")
        if data.axes[-2] == -1.0:
            self.state["move_right"] =True
        elif data.axes[-2] == 1.0:
            self.state["move_left"] =True
        elif data.axes[-1] == 1.0:
            self.state["move_up"] =True
        elif data.axes[-1] == -1.0:
            self.state["move_down"] =True
        elif data.buttons[2] == 1:
            self.state["open"] = True
        elif data.buttons[3] == 1:
            self.state["close"] = True
        elif data.buttons[1] == 1:
            self.state["reduce_pinch"] = True
        elif data.buttons[0] == 1:
            self.state["increase_pinch"] = True
        

if (__name__=="__main__"):
    dz = DeltaZ()
    dz.spin()
