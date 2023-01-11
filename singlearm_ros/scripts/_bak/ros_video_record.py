#!/usr/bin/env python
import rospy
import numpy as np
import cv2
import sys
from std_msgs.msg import String

class CaptureVideo:
	def __init__(self):
		rospy.init_node("video_capture")
		self.fps = 20.0
		self.delay = int(1.0/self.fps * 1000) # milliseconds
		self.cap = None
		self.out = None
		self.command = ''

		sub = rospy.Subscriber('record', String, self.cb)

	def open(self):
		#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
		#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 180)
		# SAVE = sys.argv[1].lower() == 'true'
		self.cap = cv2.VideoCapture(0)
		self.cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))
		if self.cap.isOpened() == False:
			print("error: cant open stream")

		frame_width = int(self.cap.get(3))
		frame_height = int(self.cap.get(4))
		self.out_width = int(frame_width/3)
		self.out_height = int(frame_height/3)

		self.out = cv2.VideoWriter(self.fname, cv2.VideoWriter_fourcc('M','P','4','V'), self.fps, (self.out_width, self.out_height))

	def cb(self, msg):
		data = msg.data
		fname, command = data.split('-')
		self.fname = fname
		self.command = command
		print(self.command)

	def close(self):
		self.cap.release()
		self.cap = None
		self.out.release()
		self.out = None
		cv2.destroyAllWindows()

	def spin(self):
		while not rospy.is_shutdown():
			# if self.out is not None and self.cap.isOpened():
			# self.open()
			# self.out = cv2.VideoWriter(fname, cv2.VideoWriter_fourcc('M','P','4','V'), self.fps, (self.out_width, self.out_height))

			if self.command == 'start':
				self.open()
				self.command = ''

			# if self.out is not None and self.cap.isOpened():
			if self.out is not None and self.cap.isOpened():
				ret, frame = self.cap.read()
				if ret == True:
					frame = cv2.resize(frame, (self.out_width, self.out_height))
					cv2.imshow('Frame', frame)
					self.out.write(frame)

					if cv2.waitKey(self.delay) == ord('q') or self.command == 'stop':
						self.close()
						# break
				else:
					self.close()
					# break

if __name__ == '__main__':
	cv = CaptureVideo()
	cv.spin()
	# cv.open()