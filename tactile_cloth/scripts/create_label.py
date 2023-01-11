import cv2
import rosbag
import rospy
import numpy as np
from cv_bridge import CvBridge
import cv2 
import time
import matplotlib.pyplot as plt
import argparse
from std_msgs.msg import Int64
import progressbar

def create_csv_from_video(bagfile, data_array, verbose = False):
    frame_count = 0
    final_data = []
    contact_type = 0
    count = 0
    total_messages = bagfile.get_message_count()
    bar = progressbar.ProgressBar(maxval=total_messages, \
    widgets=[progressbar.Bar('=', '[', ']'), ' making reskin dataset ', progressbar.Percentage()])
    bar.start()
    for topic, msg, t in bagfile.read_messages(topics=['/reskin', '/webcam_image']):
        if(count> 100000):
            break
        else:
            count = count+1
            bar.update(count)
            if(verbose):
                if(count%100 == 0):
                    print(count)
        if topic == '/reskin':
            data = np.array(msg.data) # 5 x (T x y z)
            if (frame_count >= data_array[0] and frame_count <= data_array[1]):
                contact_type = 1
            else:
                contact_type=0
            
            new_data = [data[1], data[2], data[3], data[5], data[6], data[7],
            data[9], data[10], data[11], data[13], data[14], data[15], data[17], data[18], data[19], contact_type, int(str(t))]
            final_data.append(new_data)
        if(topic == '/webcam_image'):
            frame_count = frame_count + 1

    final_data = np.array(final_data)
    bar.finish()
    return final_data

def create_data_from_classifier(bagfile, clf):
    total_messages = bagfile.get_message_count()
    bar = progressbar.ProgressBar(maxval=total_messages, \
    widgets=[progressbar.Bar('=', '[', ']'), ' making reskin dataset ', progressbar.Percentage()])
    bar.start()
    final_data=[]
    count = 0
    for topic, msg, t in bagfile.read_messages(topics=['/reskin', '/webcam_image']):
        count = count+1
        bar.update(count)
        if topic == '/reskin':
            data = np.array(msg.data) # 5 x (T x y z)
            contact_type = clf.predict([[data[1], data[2], data[3], data[5], data[6], data[7],
            data[9], data[10], data[11], data[13], data[14], data[15], data[17], data[18], data[19]]])
            new_data = [data[1], data[2], data[3], data[5], data[6], data[7],
            data[9], data[10], data[11], data[13], data[14], data[15], data[17], data[18], data[19], contact_type, int(str(t))]
            final_data.append(new_data)
    final_data = np.array(final_data)
    bar.finish()
    return final_data

def create_data_no_label(bagfile):
    total_messages = bagfile.get_message_count()
    bar = progressbar.ProgressBar(maxval=total_messages, \
    widgets=[progressbar.Bar('=', '[', ']'), ' making reskin dataset ', progressbar.Percentage()])
    bar.start()
    final_data=[]
    count = 0
    for topic, msg, t in bagfile.read_messages(topics=['/reskin', '/webcam_image']):
        count = count+1
        bar.update(count)
        if topic == '/reskin':
            data = np.array(msg.data) # 5 x (T x y z)
            new_data = [data[1], data[2], data[3], data[5], data[6], data[7],
            data[9], data[10], data[11], data[13], data[14], data[15], data[17], data[18], data[19], int(str(t))]
            final_data.append(new_data)
    final_data = np.array(final_data)
    bar.finish()
    return final_data

def label_data(data, label):
    init_data = data[:400,:]
    sqdata = np.square(init_data)
    diff_norm_data = np.abs(np.diff(np.mean(sqdata, axis = 1)))
    init_index = np.argmax(diff_norm_data)
    final_data = data[-400:,:]
    sqdata = np.square(final_data)
    diff_norm_data = np.abs(np.diff(np.mean(sqdata, axis = 1)))
    final_index = (data.shape[0] - 400)+np.argmax(diff_norm_data)
    labels = np.reshape(np.zeros(data.shape[0]),(-1,1))
    labels[init_index:final_index,0] = label
    labelled_data = np.hstack([data,labels])
    return labelled_data

def create_data_from_norm(bagfile, label):
    total_messages = bagfile.get_message_count()
    bar = progressbar.ProgressBar(maxval=total_messages, \
    widgets=[progressbar.Bar('=', '[', ']'), ' making reskin dataset ', progressbar.Percentage()])
    bar.start()
    final_data=[]
    count = 0
    for topic, msg, t in bagfile.read_messages(topics=['/reskin', '/webcam_image']):
        count = count+1
        bar.update(count)
        if topic == '/reskin':
            data = np.array(msg.data) # 5 x (T x y z)
            new_data = [data[1], data[2], data[3], data[5], data[6], data[7],
            data[9], data[10], data[11], data[13], data[14], data[15], data[17], data[18], data[19], int(str(t))]
            final_data.append(new_data)
    final_data = np.array(final_data)
    x_dat_without_time = final_data[:,:-1]
    labelled_data = label_data(x_dat_without_time, label) 
    final_data = np.hstack([labelled_data, final_data[:,-1].reshape(-1,1)])
    bar.finish()
    return final_data

if (__name__ == "__main__"):
    parser = argparse.ArgumentParser(description='Visualize a BagFile')
    parser.add_argument('--name', type=str, required=True)

    args = parser.parse_args()
    bagpath = '/home/sashank/catkin_ws/src/tactilecloth/bagfiles/'+args.name+'.bag'
    txtpath = '/home/sashank/catkin_ws/src/tactilecloth/video_markers/'+args.name+'.csv'
    filesave = '/home/sashank/catkin_ws/src/tactilecloth/csv_data/'+args.name+'.csv'
    contact_array = np.loadtxt(txtpath, delimiter=",")
    bag = rosbag.Bag(bagpath, 'r')
    final_data = create_csv_from_video(bag, contact_array, True)
    np.savetxt(filesave, final_data, delimiter=",")

    # for i in range(15):
    #     bagpath = '/home/sashank/catkin_ws/src/tactilecloth/bagfiles/0cloth_7Feb/'+str(i+1)+'.bag'
    #     txtpath = '/home/sashank/catkin_ws/src/tactilecloth/classification_data/0cloth_7feb/'+str(i+1)+'/'+str(i+1)+'_markers.csv'
    #     filesave = '/home/sashank/catkin_ws/src/tactilecloth/classification_data/0cloth_7feb/'+str(i+1)+'/'+str(i+1)+'_reskin_data.csv'
    #     contact_array = np.loadtxt(txtpath, delimiter=",")
    #     bag = rosbag.Bag(bagpath, 'r')
    #     final_data = create_csv_from_video(bag, contact_array, False)
    #     np.savetxt(filesave, final_data, delimiter=",")
    #     pass   
    # pass