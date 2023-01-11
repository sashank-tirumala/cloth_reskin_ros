from curses.ascii import NUL
import linear_actuator_pb2
import numpy as np
from serial import Serial
from math import *
import time

max_circle_radius = 2


def create_joint_positions(val):
    for i in range(12):
        if i == ind:
            delta_message.joint_pos.append(val)
        else:
            delta_message.joint_pos.append(0.03)


def inv_kin(x, y, z):
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
    print("in_func:", z1, z2, z3)
    z1, z2, z3 = np.clip([z1, z2, z3], 0.02, 9.98)
    return [z1, z2, z3]
