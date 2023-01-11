from curses.ascii import NUL
import linear_actuator_pb2
import numpy as np
from serial import Serial
from math import *
import time
arduino = Serial('/dev/ttyACM0', 57600)  


delta_message = linear_actuator_pb2.lin_actuator()
delta_message.id = 1

NUM_MOTORS = 6

def create_joint_positions(val):
    for i in range(12):
        if i<NUM_MOTORS:
            delta_message.joint_pos.append(val)
        else:
            delta_message.joint_pos.append(0.0)

while True:
    x = input("Position in cm? ")
    create_joint_positions(float(x)/100)

    print(delta_message)

    serialized = delta_message.SerializeToString()
    arduino.write(bytes('<', 'utf-8') + serialized + bytes('>', 'utf-8'))
    reachedPos = str(arduino.readline())
    while reachedPos[0]!="~": 
        print(reachedPos[0], reachedPos[0]!="~")
        reachedPos = str(arduino.readline().decode())
    delta_message.Clear()
