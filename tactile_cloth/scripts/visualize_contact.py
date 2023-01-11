import cv2
import rosbag
import numpy as np
from cv_bridge import CvBridge
import cv2 
import time
import matplotlib.pyplot as plt
import argparse
import progressbar
def create_video_array(bagfile, data_arr, CUTOFF, PTS_PLOTTED = 250, verbose = False):
    bridge = CvBridge()
    images = []
    bx = []
    by = []
    bz = []
    ct = []
    plot_images=[]
    fig = plt.figure()
    count = 0
    final_images=[]
    time_vals = data_arr[:,-1]
    total_messages = bagfile.get_message_count('/webcam_image')
    bar = progressbar.ProgressBar(maxval=total_messages, \
    widgets=[progressbar.Bar('=', '[', ']'), ' making contact video ', progressbar.Percentage()])
    bar.start()
    for topic, msg, t in bagfile.read_messages(topics=['/webcam_image']):
        if topic == '/webcam_image':
            img = bridge.imgmsg_to_cv2(msg)
            images.append(img)
            nearest_greater_index = np.searchsorted(time_vals,[int(str(t)),],side='right')[0]
            earliest_index = nearest_greater_index - PTS_PLOTTED if nearest_greater_index - PTS_PLOTTED > 0 else 0
            current_data = data_arr[earliest_index:nearest_greater_index, :]
            contact_data = current_data[:,-2]
            max_cont = np.max(data_arr[:,-2])
            plt.cla()
            plt.xlim([0, 550])
            plt.ylim([-0.1, max_cont+1])
            plt.title("Contact Signal")
            plt.plot(contact_data, 'r')
            fig.canvas.draw()
            img1 = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
                sep='')
            img1  = img1.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            img1 = cv2.cvtColor(img1,cv2.COLOR_RGB2BGR)
            plot_images.append(img1)
            # store imgs in data structure
            new_img = np.concatenate((img, img1))
            final_images.append(new_img)
            count = count+1
            bar.update(count)
            if(count%10== 0):
                if(verbose):
                    print(count)
                pass
    bar.finish()
    return final_images
   
if(__name__ == "__main__"):
    # parser = argparse.ArgumentParser(description='Visualize a BagFile')
    # parser.add_argument('--name', type=str, required=True)
    # args = parser.parse_args()
    bagpath = "/home/sashank/catkin_ws/src/tactilecloth/bagfiles/0cloth_21Jan/13.bag"
    bag = rosbag.Bag(bagpath)
    csv_path = "/home/sashank/catkin_ws/src/tactilecloth/classification_data/0cloth_21Jan/13/13_reskin_data_with_classification.csv"
    data_arr = np.loadtxt(csv_path, delimiter=",")
    CUTOFF = 10000
    finale_video = create_video_array(bag, data_arr, CUTOFF, 500, False)
    height,width,layers=finale_video[1].shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video=cv2.VideoWriter('/home/sashank/catkin_ws/src/tactilecloth/videos/test_contact.avi',fourcc,10,(width,height))

    for j in range(len(finale_video)):
        video.write(finale_video[j])

    cv2.destroyAllWindows()
    video.release()
    
    pass