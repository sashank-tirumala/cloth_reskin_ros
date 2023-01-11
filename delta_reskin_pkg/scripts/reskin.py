#!/usr/bin/python
import rospy
from reskin_sensor import ReSkinBase
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
import numpy as np

BTN_X_INDEX = 2


class ReSkin:
    def __init__(self):
        rospy.init_node("reskin")
        self.pub = rospy.Publisher("/reskin", Float64MultiArray, queue_size=10)
        self.sensor = ReSkinBase(
            num_mags=5,
            port="/dev/ttyACM0",
            baudrate=115200,
            burst_mode=True,
            device_id=1,
        )
        self.rate = rospy.Rate(1)

    def spin(self):
        while not rospy.is_shutdown():
            data = self.sensor.get_data(num_samples=1)
            data = data[0].data
            if np.count_nonzero(data == 0) > 5 or (data[0] > 40 or data[0] < 10):
                rospy.logerr(
                    "Reskin is not working, please unplug/replug or reupload the arduino code"
                )
                rospy.logerr(data)

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
            self.pub.publish(msg)


if __name__ == "__main__":
    rk = ReSkin()
    rk.spin()
