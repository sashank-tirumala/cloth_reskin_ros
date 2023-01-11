#!/usr/bin/python
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Joy
import numpy as np
from serial import Serial
from lin_deltaz_utils import move_to_xyz_pos
import numpy as np
import linear_actuator_pb2
from time import sleep
from singlearm_ros.msg import MoveFingersAction, MoveFingersActionResult
import actionlib


class DeltaZ:
    def __init__(self):
        rospy.init_node("linear_delta_joystick_node")
        self.joy_sub = rospy.Subscriber("/joy", Joy, self.joy_callback)
        self.arduino = Serial("/dev/ttyACM1", 57600)
        self.delta_message = linear_actuator_pb2.lin_actuator()
        self.delta_message.id = 1
        self.cur_pos = np.array([0.0, 0.0, 3.0, 0.0, 0.0, 3.0])
        self.vel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.x_vel_under = 0
        self.z_vel_under = 0
        self.x_vel_over = 0
        self.z_vel_over = 0
        self.constant_speed_factor = 0.1

    def joy_callback(self, data):
        self.x_vel_under = float(data.axes[0] * self.constant_speed_factor)
        self.z_vel_under = float(data.axes[1] * self.constant_speed_factor)
        self.x_vel_over = float(data.axes[2] * self.constant_speed_factor)
        self.z_vel_over = float(data.axes[3] * self.constant_speed_factor)

    def continuous_loop(self):
        while not rospy.is_shutdown():
            self.vel[0] = self.x_vel_under
            self.vel[2] = self.z_vel_under
            self.vel[3] = self.x_vel_over
            self.vel[5] = self.z_vel_over
            self.cur_pos = self.cur_pos + self.vel
            move_to_xyz_pos(self.cur_pos, self.delta_message, self.arduino)


if __name__ == "__main__":
    dz = DeltaZ()
    dz.continuous_loop()
