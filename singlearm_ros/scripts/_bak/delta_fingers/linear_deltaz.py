#!/usr/bin/env python

from curses.ascii import NUL
import linear_actuator_pb2
import numpy as np
from serial import Serial
from math import *
import time

class LinearDelta:
    def __init__(self):
        self.max_circle_radius = 2
        self.arduino = Serial('/dev/ttyACM0', 57600)  
        self.delta_message = linear_actuator_pb2.lin_actuator()
        self.delta_message.id = 1
        self.ind = 4
        self.NUM_MOTORS = 6

    def inv_kin_left(self, x, y, z):
        self.max_circle_radius = 2
        center_dist = np.sqrt(x**2 + y**2)
        if(center_dist<self.max_circle_radius):
            pass
        else:
            x = x*self.max_circle_radius/np.sqrt(x**2 + y**2)
            y = y*self.max_circle_radius/np.sqrt(x**2 + y**2)
        z = z+6
        a = 1
        b = -2*z
        c = z**2 -36 + (-1-x)**2 + (0-y)**2
        z1 = (-b - np.sqrt(b**2 -  4*a*c))/(2*a)

        c = z**2 -36 + (0.5-x)**2 + (0.876-y)**2
        z2 = (-b - np.sqrt(b**2 -  4*a*c))/(2*a)

        c = z**2 -36 + (0.5-x)**2 + (-0.876-y)**2
        z3 = (-b - np.sqrt(b**2 -  4*a*c))/(2*a)
        z1,z2,z3 = np.clip([z1,z2,z3], 0.02, 9.98)
        z1,z2,z3 = np.array([z1,z2,z3])/100.0
        return[z1,z2,z3]

    def inv_kin_right(self, x, y, z):
        max_circle_radius = 2
        center_dist = np.sqrt(x**2 + y**2)
        if(center_dist<max_circle_radius):
            pass
        else:
            x = x*max_circle_radius/np.sqrt(x**2 + y**2)
            y = y*max_circle_radius/np.sqrt(x**2 + y**2)
        z = z+6
        a = 1
        b = -2*z
        c = z**2 -36 + (1-x)**2 + (0-y)**2
        z1 = (-b - np.sqrt(b**2 -  4*a*c))/(2*a)

        c = z**2 -36 + (-0.5-x)**2 + (-0.876-y)**2
        z2 = (-b - np.sqrt(b**2 -  4*a*c))/(2*a)

        c = z**2 -36 + (-0.5-x)**2 + (0.876-y)**2
        z3 = (-b - np.sqrt(b**2 -  4*a*c))/(2*a)
        z1,z2,z3 = np.clip([z1,z2,z3], 0.02, 9.98)
        z1,z2,z3 = np.array([z1,z2,z3])/100.0
        return[z1,z2,z3]

    def create_joint_positions(self, vals):
        for i in range(12):
            if i<len(vals):
                delta_message.joint_pos.append(np.clip(vals[i], 0.02/100.0, 9.98/100.0))
            else:
                delta_message.joint_pos.append(0.03)

    def move_to_use_pos(self):
        while True:
            x = input("x-coord in cm? ")
            y = input("x-coord in cm? ")
            z = input("x-coord in cm? ")

            z1,z2,z3 = inv_kin_left(x,y,z)
            z4,z5,z6 = inv_kin_right(x,y,z)
            print("z_vals: ",z1,z2,z3,z4,z5,z6)
            create_joint_positions(np.array([z1,z2,z3,z4,z5,z6]))

            print(delta_message)

            serialized = delta_message.SerializeToString()
            arduino.write('<'+ serialized + '>')
            reachedPos = str(arduino.readline())
            while reachedPos[0]!="~": 
                print(reachedPos)
                reachedPos = str(arduino.readline().decode())
            delta_message.Clear()
            
    def move_in_circle(self):
        radius = 2
        omega = 2*np.pi*0.05 # the problem was with resolution 
        theta = 0
        while True:
            x = radius*np.cos(theta)
            y = radius*np.sin(theta)
            z= 5
            z1,z2,z3 = inv_kin_left(x,y,z)
            z4,z5,z6 = inv_kin_right(x,y,z)
            print("z_vals: ",z1,z2,z3,z4,z5,z6)
            create_joint_positions(np.array([z1,z2,z3,z4,z5,z6]))

            # print(delta_message)

            serialized = delta_message.SerializeToString()
            arduino.write('<'+ serialized + '>')
            reachedPos = str(arduino.readline())
            while reachedPos[0]!="~": 
                # print(reachedPos)
                reachedPos = str(arduino.readline().decode())
            delta_message.Clear()
            theta = theta+omega

# if __name__ == '__main__':
#     move_to_use_pos()
#     move_in_circle()