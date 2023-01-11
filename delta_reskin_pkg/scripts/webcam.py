#!/usr/bin/python
import argparse
import rospy
import cv2
import os
import sys
import time
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

BTN_X_INDEX = 2

# publishes camera data
class WebCam:
    def __init__(self, webcam_idx):
        name = "webcam_image"

        rospy.init_node(name + "_node")
        self.bridge = CvBridge()

        self.pub = rospy.Publisher("/" + name, Image, queue_size=100)
        self.sub = rospy.Subscriber(
            "dbg_image_record", String, callback=self.set_record
        )
        self.is_recording = False
        self.fourcc = cv2.VideoWriter_fourcc(*"MP4V")
        self.frame = None
        self.video = None
        self.cap = cv2.VideoCapture(webcam_idx)
        if self.cap.isOpened() == False:
            print("Error opening video stream or file for %s" % name)
            sys.exit(0)
        self.shutdown = rospy.on_shutdown(self.process_shutdown)

    def spin(self):
        while not rospy.is_shutdown():
            ret, frame = self.cap.read()
            self.frame = frame
            try:
                msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            except Exception as e:
                rospy.logerr(e)
                break

            self.pub.publish(msg)

            # Save if recording
            if self.is_recording:
                self.video.write(frame)

    def set_record(self, msg):
        if msg.data == "start_dbg_record":
            self.trial_type_dir = rospy.get_param("trial_type_dir")

            height, width, _ = self.frame.shape
            ts = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            self.mp4_path = "%s/%s_recording.mp4" % (self.trial_type_dir, ts)
            self.video = cv2.VideoWriter(
                self.mp4_path, self.fourcc, 10, (width, height)
            )
            self.is_recording = True
        elif msg.data == "end_dbg_record":
            self.is_recording = False
            self.video.release()

    def process_shutdown(self):
        if self.video is not None:
            self.video.release()


if __name__ == "__main__":
    wc = WebCam(0)
    wc.spin()
