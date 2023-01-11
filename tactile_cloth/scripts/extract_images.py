import cv2
import os
from cv_bridge import CvBridge
import rosbag
import argparse

def extract_pred_images(bagfile, save_dir):
    """
    This function extracts classifier input images from a bagfile and saves them to save_dir
    """
    bag = rosbag.Bag(bagfile)
    bridge = CvBridge()
    topics = ['/webcam_image', '/classifier_commands']
    predicting = False
    count = 0
    for topic, msg, t in bag.read_messages(topics=topics):
        if topic == '/classifier_commands':
            if msg.data == 'start':
                predicting = True
            elif msg.data == 'end':
                predicting = False

        if topic == '/webcam_image':
            if predicting:
                print(t)
                img = bridge.imgmsg_to_cv2(msg)
                img = img[120:, 280:, :] # crop for classifier
                cv2.imwrite("%s/%d.png" % (save_dir, count), img)
                count += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create Complete DataSet')
    parser.add_argument('--dir', help="directory containing bagfile", 
        type=str, default=None)
    args, _ = parser.parse_known_args()

    bagdir = "%s/data.bag" % args.dir
    imgdir = "%s/input_images" % args.dir
    if not os.path.exists(imgdir):
        os.makedirs(imgdir)
    extract_pred_images(bagfile=bagdir, save_dir=imgdir)