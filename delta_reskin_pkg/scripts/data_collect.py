import cv2
import rosbag
import numpy as np
from cv_bridge import CvBridge
import cv2
import time
import matplotlib.pyplot as plt
import argparse
import progressbar
import pickle


def get_data_from_folder(bag_path, csv_path):
    bag = rosbag.Bag(bag_path)
    data_arr = np.loadtxt(csv_path, delimiter=",")
    time_vals = data_arr[:, -1]
    bridge = CvBridge()
    final_data = []
    i = 0
    for topic, msg, t in bag.read_messages(topics=["/webcam_image"]):
        if topic == "/webcam_image":
            current_img = bridge.imgmsg_to_cv2(msg)
            nearest_greater_index = np.searchsorted(
                time_vals, [int(str(t))], side="right"
            )[0]
            current_reskin_data = data_arr[
                nearest_greater_index - 17 : nearest_greater_index, :
            ]  # An approximation
            final_data.append([i, current_img, current_reskin_data])
            i = i + 1
    return final_data


if __name__ == "__main__":
    CLASSIFICATION_FOLDER = (
        "/home/sashank/catkin_ws/src/tactilecloth/classification_data/"
    )  # Change this to the path in your system
    FOLDER_NAMES = [
        "0cloth_21Jan/",
        "1cloth_21Jan",
    ]  # The only folders with the new data
    index = 3
    bagpath = (
        CLASSIFICATION_FOLDER + FOLDER_NAMES[0] + str(index) + "/" + str(index) + ".bag"
    )
    csvpath = (
        CLASSIFICATION_FOLDER
        + FOLDER_NAMES[0]
        + str(index)
        + "/"
        + str(index)
        + "_reskin_data_with_classification.csv"
    )
    data = get_data_from_folder(bagpath, csvpath)
    print(data[1], data[12], data[22])

    open_file = open("3_0CLOTH.pkl", "wb")
    pickle.dump(data, open_file)
    open_file.close()
    pass
