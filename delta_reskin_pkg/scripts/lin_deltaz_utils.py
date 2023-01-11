from curses.ascii import NUL
import linear_actuator_pb2
import numpy as np
from math import *
import time
import rospy
import actionlib
from std_msgs.msg import String
from time import sleep

max_circle_radius = 2


delta_message = linear_actuator_pb2.lin_actuator()
delta_message.id = 1
ind = 4
NUM_MOTORS = 6


def inv_kin_left(x, y, z):
    max_circle_radius = 2
    center_dist = np.sqrt(x ** 2 + y ** 2)
    if center_dist < max_circle_radius:
        pass
    else:
        x = x * max_circle_radius / np.sqrt(x ** 2 + y ** 2)
        y = y * max_circle_radius / np.sqrt(x ** 2 + y ** 2)
    z = z + 6
    a = 1
    b = -2 * z
    c = z ** 2 - 36 + (-1 - x) ** 2 + (0 - y) ** 2
    z1 = (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)

    c = z ** 2 - 36 + (0.5 - x) ** 2 + (0.876 - y) ** 2
    z2 = (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)

    c = z ** 2 - 36 + (0.5 - x) ** 2 + (-0.876 - y) ** 2
    z3 = (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    z1, z2, z3 = np.clip([z1, z2, z3], 0.02, 9.98)
    z1, z2, z3 = np.array([z1, z2, z3]) / 100.0
    return [z1, z2, z3]


def inv_kin_right(x, y, z):
    max_circle_radius = 2
    center_dist = np.sqrt(x ** 2 + y ** 2)
    if center_dist < max_circle_radius:
        pass
    else:
        x = x * max_circle_radius / np.sqrt(x ** 2 + y ** 2)
        y = y * max_circle_radius / np.sqrt(x ** 2 + y ** 2)
    z = z + 6
    a = 1
    b = -2 * z
    c = z ** 2 - 36 + (1 - x) ** 2 + (0 - y) ** 2
    z1 = (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)

    c = z ** 2 - 36 + (-0.5 - x) ** 2 + (-0.876 - y) ** 2
    z2 = (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)

    c = z ** 2 - 36 + (-0.5 - x) ** 2 + (0.876 - y) ** 2
    z3 = (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    z1, z2, z3 = np.clip([z1, z2, z3], 0.02, 9.98)
    z1, z2, z3 = np.array([z1, z2, z3]) / 100.0
    return [z1, z2, z3]


def create_joint_positions(vals, delta_message):
    for i in range(12):
        if i < len(vals):
            delta_message.joint_pos.append(np.clip(vals[i], 0.02 / 100.0, 9.98 / 100.0))
        else:
            delta_message.joint_pos.append(0.03)


def move_to_use_pos(delta_message, arduino):
    while True:
        x1 = input("x-coord in cm? ")
        y1 = input("y-coord in cm? ")
        z1 = input("z-coord in cm? ")
        x2 = input("x-coord in cm? ")
        y2 = input("y-coord in cm? ")
        z2 = input("z-coord in cm? ")

        z1, z2, z3 = inv_kin_left(x, y, z)
        z4, z5, z6 = inv_kin_right(-x, y, z)
        create_joint_positions(np.array([z1, z2, z3, z4, z5, z6]), delta_message)
        serialized = delta_message.SerializeToString()
        arduino.write(b"\xa6~~" + serialized + b"\xa7~~")
        reachedPos = str(arduino.readline())
        while reachedPos[0] != "~":
            reachedPos = str(arduino.readline().decode())
        delta_message.Clear()


def move_to_xyz_pos(xyz_pos, delta_message, arduino):
    x_left, y_left, z_left = xyz_pos[0:3]
    x_right, y_right, z_right = xyz_pos[3:6]
    z1, z2, z3 = inv_kin_left(x_left, y_left, z_left)
    z4, z5, z6 = inv_kin_right(x_right, y_right, z_right)
    create_joint_positions(np.array([z1, z2, z3, z4, z5, z6]), delta_message)
    serialized = delta_message.SerializeToString()
    rospy.loginfo("here")
    arduino.write(bytes(b"\xa6~~") + serialized + bytes(b"\xa7~~"))
    delta_message.Clear()


def move_in_circle():
    radius = 2
    omega = 2 * np.pi * 0.05
    theta = 0
    while True:
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = 5
        z1, z2, z3 = inv_kin_left(x, y, z)
        z4, z5, z6 = inv_kin_right(x, y, z)
        print("z_vals: ", z1, z2, z3, z4, z5, z6)
        create_joint_positions(np.array([z1, z2, z3, z4, z5, z6]))
        serialized = delta_message.SerializeToString()
        arduino.write("<" + serialized + ">")
        reachedPos = str(arduino.readline())
        while reachedPos[0] != "~":
            reachedPos = str(arduino.readline().decode())
        delta_message.Clear()
        theta = theta + omega


if __name__ == "__main__":
    arduino = Serial("/dev/ttyACM0", 57600)
    delta_message = linear_actuator_pb2.lin_actuator()
    delta_message.id = 1
    y_offset = 0.17
    pinch = 2
    xyz_pos = np.array([-pinch, y_offset, 4, 0, 0, 4])
    move_to_xyz_pos(xyz_pos, delta_message, arduino)
