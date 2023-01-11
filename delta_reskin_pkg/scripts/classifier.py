#!/usr/bin/python
import rospy
import pickle
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
from std_msgs.msg import String
from std_msgs.msg import Int64
from sklearn import neighbors
import numpy as np
import actionlib
from joblib import load

FLAG_ALWAYS_CLASSIFY = False


class Classifier:
    def __init__(self):
        rospy.init_node("classifier")
        self.clf_dir_name = rospy.get_param("/app/clf_dir_name")
        if self.clf_dir_name == "":
            return

        self.clf = load(self.clf_dir_name + "/classifier.joblib")
        self.scaler = load(self.clf_dir_name + "/scaler.joblib")
        print("model loaded successfully: ", self.clf)

        self.lin_sub = rospy.Subscriber(
            "/classifier_commands", String, self.clf_callback
        )

        self.reskin_sub = rospy.Subscriber(
            "/reskin", Float64MultiArray, self.reskin_callback
        )
        self.reskin_data = np.zeros(15)

        self.pub = rospy.Publisher("/classifier_output", Int64, queue_size=10)

        self.FLAG_classify = False

    def clf_callback(self, msg):
        rospy.loginfo("callback: %s" % msg.data)
        if msg.data == "start":
            self.FLAG_classify = True
        elif msg.data == "end":
            self.FLAG_classify = False

    def spin(self):
        while not rospy.is_shutdown():
            if FLAG_ALWAYS_CLASSIFY:
                reskin_dat = np.array(
                    self.reskin_data
                )  # Automatically makes a copy, fastest way for small data
                reskin_dat = reskin_dat.reshape(-1, 15)
                reskin_dat = self.scaler.transform(reskin_dat)
                y_pred = self.clf.predict(reskin_dat) - 1  # subtract to get num layers
                self.pub.publish(Int64(int(np.around(np.mean(y_pred)))))
                self.final_y = int(np.around(np.mean(y_pred)))
            else:
                if self.FLAG_classify:
                    reskin_dat = np.array(
                        self.reskin_data
                    )  # Automatically makes a copy, fastest way for small data
                    reskin_dat = reskin_dat.reshape(-1, 15)
                    reskin_dat = self.scaler.transform(reskin_dat)
                    y_pred = (
                        self.clf.predict(reskin_dat) - 1
                    )  # subtract to get num layers
                    self.pub.publish(Int64(int(np.around(np.mean(y_pred)))))
                    self.final_y = int(np.around(np.mean(y_pred)))

    def reskin_callback(self, data):
        cdat = data.data
        cnumpy = np.array(
            [
                cdat[1],
                cdat[2],
                cdat[3],
                cdat[5],
                cdat[6],
                cdat[7],
                cdat[9],
                cdat[10],
                cdat[11],
                cdat[13],
                cdat[14],
                cdat[15],
                cdat[17],
                cdat[18],
                cdat[19],
            ]
        )
        self.reskin_data = cnumpy[np.newaxis, :]


if __name__ == "__main__":
    clf = Classifier()
    clf.spin()
