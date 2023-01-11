#!/usr/bin/python
import rospy
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
from sensor_msgs.msg import Joy
# from inputs import get_gamepad
import numpy as np
import serial
BTN_Y_INDEX = 3
BTN_B_INDEX = 1
TRAJ_SPEED = 1
TRAJ_LENGTH = 40
class DeltaZ:
    def __init__(self):
        rospy.init_node('deltaz')
        self.publish_deltaZ = False
        self.vertical_control = False
        self.joy_sub = rospy.Subscriber("/joy", Joy, self.joy_callback)
        self.ser = serial.Serial()
        port = "/dev/ttyUSB0"
        self.ser.port = port
        self.ser.baudrate = 9600
        self.ser.open()
        self.ser2 = serial.Serial()
        port2 = "/dev/ttyUSB1"
        self.ser2.port = port2
        self.ser2.baudrate = 9600
        self.ser2.open()
        self.joy_data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.pinch = False
        self.traj_counter = 0
        pass

    def spin(self):
        count = 0
        current_pos1 = np.array([0.,0.,-55.])
        current_pos2 = np.array([0.,0.,-55.])
        while not rospy.is_shutdown():
            # print(self.vertical_control)
            rx,ry,rz,lx,ly,lz = self.joy_data
            lx= lx
            ly= -ly
            ry = -ry
            rx = rx
            lz = -lz
            rz = -  rz
            vel=2
            current_pos1[0] = current_pos1[0] + rx*vel
            if(self.vertical_control):
                current_pos1[2] = current_pos1[2]-ry*vel
                current_pos1[2] = np.clip(current_pos1[2], -75, -35)
            else:
                current_pos1[1] = current_pos1[1] + ry*vel
            new_rad = (current_pos1[0]**2 + current_pos1[1]**2)**0.5
            new_rad = np.clip(new_rad, -1, 30)
            theta = np.arctan2(current_pos1[1], current_pos1[0])
            current_pos1[0] = new_rad*np.cos(theta)
            current_pos1[1] = new_rad*np.sin(theta)
            # print(command)
            current_pos2[0] = current_pos2[0] + lx*vel
            
            if(self.vertical_control):
                current_pos2[2] = current_pos2[2]-ly*vel
                current_pos2[2] = np.clip(current_pos2[2], -75, -35)
            else:
                current_pos2[1] = current_pos2[1] + ly*vel
            new_rad = (current_pos2[0]**2 + current_pos2[1]**2)**0.5
            new_rad = np.clip(new_rad, -1, 30)
            theta = np.arctan2(current_pos2[1], current_pos2[0])
            current_pos2[0] = new_rad*np.cos(theta)
            current_pos2[1] = new_rad*np.sin(theta)
            if(self.pinch == True):
                if (self.traj_counter == 0):
                    current_pos1[0] = 0
                    current_pos1[1] = 0
                    current_pos1[2] = -55
                    
                    current_pos2[0] = 0
                    current_pos2[1] = 0
                    current_pos2[2] = -55
                    self.traj_counter = self.traj_counter + 1

                if(self.traj_counter > 0 and self.traj_counter < TRAJ_LENGTH):
                    current_pos1[0] = current_pos1[0] + TRAJ_SPEED
                    current_pos2[0] = current_pos2[0] + -1*TRAJ_SPEED
                    self.traj_counter = self.traj_counter + 1

                elif(self.traj_counter>= TRAJ_LENGTH and self.traj_counter < 2*TRAJ_LENGTH):
                    current_pos1[0] = current_pos1[0] + -1*TRAJ_SPEED
                    current_pos2[0] = current_pos2[0] + TRAJ_SPEED
                    self.traj_counter = self.traj_counter + 1
                
                else:
                    self.traj_counter = 0
                    self.pinch=False

                pass
            command = "GOTO "+str(current_pos1[0])+","+str(current_pos1[1])+","+str(current_pos1[2])+"\n"
            command2 = "GOTO "+str(current_pos2[0])+","+str(current_pos2[1])+","+str(current_pos2[2])+"\n"
            command = command.encode('utf-8')
            command2 = command2.encode('utf-8')
            # print(command2)
            self.ser2.write(command) 
            self.ser.write(command2)
            count = count+1

        pass

    def joy_callback(self,data):
        if(data.buttons[BTN_Y_INDEX] == 1):
            self.vertical_control = not self.vertical_control
        if(data.buttons[BTN_B_INDEX] == 1):
            self.pinch = True
            self.traj_counter = 0
        self.joy_data = data.axes[:6]
        pass

if (__name__=="__main__"):
    dz = DeltaZ()
    dz.spin()
