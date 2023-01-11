#!/usr/bin/python
import rospy
from reskin_sensor import ReSkinBase
# from rospy.topics import Publisher 
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
import numpy as np
# from sensor_msgs.msg import Joy
BTN_X_INDEX = 2
class ReSkin:
    def __init__(self):
        rospy.init_node('reskin')
        self.pub = rospy.Publisher('/reskin', Float64MultiArray, queue_size=10)
        self.sensor = ReSkinBase(
        num_mags=5,
        port="/dev/ttyACM0",
        baudrate=115200,
        burst_mode=True,
        device_id=1)
        self.rate = rospy.Rate(1)
        # self.publish_reskin = False
        # self.joy_sub = rospy.Subscriber("/joy", Joy, self.joy_callback)
    
    def spin(self):
        while not rospy.is_shutdown():
            data = self.sensor.get_data(num_samples=1)
            data = data[0].data
            if np.count_nonzero(data == 0) > 5 or (data[0] > 40 or data[0] < 10):
                rospy.logerr("Reskin is not working, please unplug/replug or reupload the arduino code")
                rospy.logerr(data)
                # rospy.signal_shutdown()

            new_data = []
            for i in range(len(data)):
                new_data.append(data[i])
            msg = Float64MultiArray()
            msg.data = new_data
            dim = MultiArrayDimension()
            dim.size = len(msg.data)
            dim.label = "command"
            dim.stride = len(msg.data)
            msg.layout.dim.append(dim)
            msg.layout.data_offset = 0
            # if(self.publish_reskin):
            #     self.pub.publish(msg)
            self.pub.publish(msg)
    
    # def joy_callback(self,data):
    #     if(data.buttons[BTN_X_INDEX] == 1):
    #         self.publish_reskin = not self.publish_reskin 

    #     pass

if __name__ == '__main__':
    rk = ReSkin()
    rk.spin()