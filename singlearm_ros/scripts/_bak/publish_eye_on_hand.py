#!/usr/bin/env python

import rospy
import numpy as np
import copy
import tf
import tf.transformations as tfs
import geometry_msgs.msg
from easy_handeye.handeye_calibration import HandeyeCalibration


rospy.init_node('handeye_calibration_publisher')
while rospy.get_time() == 0.0:
    pass

# Get calibration from camcolor to panda link7
cal = HandeyeCalibration.from_file('easy_handeye_eye_on_hand')
cal_tf = cal.transformation.transform
cal_q = np.array([cal_tf.rotation.x, cal_tf.rotation.y, cal_tf.rotation.z, cal_tf.rotation.w])
T_panda2camcolor = tfs.quaternion_matrix(cal_q)
T_panda2camcolor[:3, 3] = np.array([cal_tf.translation.x, cal_tf.translation.y, cal_tf.translation.z])

# get transform from camcolor to cambase
listener = tf.TransformListener()
listener.waitForTransform("camera_color_optical_frame", "camera_link", rospy.Time(0), rospy.Duration(3.0))
(trans,rot) = listener.lookupTransform('camera_color_optical_frame', 'camera_link', rospy.Time(0))

T_camcolor2cambase = tfs.quaternion_matrix(rot)
T_camcolor2cambase[:3, 3] = trans
print(T_camcolor2cambase)

# get transform from panda to cambase
T_panda2cambase = T_panda2camcolor.dot(T_camcolor2cambase)
q_panda2cambase = tfs.quaternion_from_matrix(T_panda2cambase)
print(q_panda2cambase)

br = tf.TransformBroadcaster()
rate = rospy.Rate(100.0)
translation = np.array([T_panda2cambase[0, 3], T_panda2cambase[1, 3], T_panda2cambase[2, 3]])
rotation = q_panda2cambase

while not rospy.is_shutdown():
    # br.sendTransform(translation, rotation, rospy.Time.now(), 'camera_link', 'panda_2_hand')
    br.sendTransform(translation, rotation, rospy.Time.now(), 'camera_link', '/panda_1/panda_hand')
    rate.sleep()

# msg = geometry_msgs.msg.TransformStamped()
# msg.header.frame_id = 'panda_2_link7'
# msg.child_frame_id = 'camera_link'
# msg.transform.translation.x = T_panda2cambase[0, 3]
# msg.transform.translation.y = T_panda2cambase[1, 3]
# msg.transform.translation.z = T_panda2cambase[2, 3]
# msg.transform.rotation.x = q_panda2cambase[0]
# msg.transform.rotation.y = q_panda2cambase[1]
# msg.transform.rotation.z = q_panda2cambase[2]
# msg.transform.rotation.w = q_panda2cambase[3]
