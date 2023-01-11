import cv2
import rosbag
import numpy as np
from cv_bridge import CvBridge
import cv2 
import time
import argparse
import progressbar
import matplotlib.pyplot as plt
from collections import deque

def plot_all_data(bagfile, fname, has_prediction=False):
    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, sharex=True)
    axes = [ax1, ax2, ax3, ax4, ax5, ax6]
    pred = []
    topics = ['/reskin', '/webcam_image', '/classifier_output', '/classifier_commands'] if has_prediction else ['/reskin', '/webcam_image']
    
    init_time = None
    delta_time = None
    reskin_deque = deque()
    clf_deque = deque()
    last_reskin_dmsecs = None
    clf_starts = []
    clf_stops = []

    for topic, msg, t in bagfile.read_messages(topics=topics):
        if init_time is None:
            init_time = t
        time_diff = (t - init_time)
        delta_msecs = time_diff.secs + time_diff.nsecs / 1e9

        if topic == '/classifier_commands':
            if msg.data =='start':
                clf_starts.append(delta_msecs)
                print("start")
                print(delta_msecs)
            elif msg.data == 'end':
                print("stop")
                print(delta_msecs)
                clf_stops.append(delta_msecs)

        if topic == '/reskin':
            data = np.array(msg.data) # 5 x (T x y z)

            magns = np.array([
                [data[1], data[2], data[3]],
                [data[5], data[6], data[7]],
                [data[9], data[10], data[11]],
                [data[13], data[14], data[15]],
                [data[17], data[18], data[19]],
            ])

            reskin_deque.append((delta_msecs, magns))
            
        elif topic == '/classifier_output':
            clf_deque.append((delta_msecs, msg.data))

    ax1.set_ylabel("C")
    ax2.set_ylabel("T")
    ax3.set_ylabel("R")
    ax4.set_ylabel("B")
    ax5.set_ylabel("L")
    ax6.set_ylabel("Pred")
    reskin_dmsecs = np.array([x[0] for x in reskin_deque]) # t x 1
    reskin_magns = np.array([x[1] for x in reskin_deque]) # t x 5 x 3
    
    clf_dmsecs = np.array([x[0] for x in clf_deque])
    clf_preds = np.array([x[1] for x in clf_deque])

    for i in range(5):
        axes[i].plot(reskin_dmsecs, reskin_magns[:, i, 0], 'r')
        axes[i].plot(reskin_dmsecs, reskin_magns[:, i, 1], 'g')
        axes[i].plot(reskin_dmsecs, reskin_magns[:, i, 2], 'b')

    for t in clf_starts:
        axes[5].axvline(t, c='blue')
    for t in clf_stops:
        axes[5].axvline(t, c='red')
    
    axes[5].set_ylim(-1.5, 3.5)
    axes[5].plot(clf_dmsecs, clf_preds)
    axes[5].set_yticks([-1, 0, 1, 2, 3])
    axes[5].set_xticklabels(["-1", "0", "1", "2", "3"], fontsize=11)
    plt.savefig(fname)


def create_video_array(bagfile, CUTOFF, has_prediction = False, plot_progress=False):
    bridge = CvBridge()
    images = []
    magn1 = []
    magn2 = []
    magn3 = []
    magn4= []
    magn5 = []
    
    pred = []
    
    plot_images=[]
    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, sharex=True)
    
    axes = [ax1, ax2, ax3, ax4, ax5, ax6]
    count = 0
    final_images=[]
    if plot_progress:
        total_messages = bagfile.get_message_count()
        bar = progressbar.ProgressBar(maxval=total_messages, \
        widgets=[progressbar.Bar('=', '[', ']'), ' making magnetometers video ', progressbar.Percentage()])
        bar.start()
    topics = ['/reskin', '/webcam_image', '/classifier_output'] if has_prediction else ['/reskin', '/webcam_image']
    
    init_time = None
    delta_time = None
    reskin_deque = deque()
    clf_deque = deque()
    last_reskin_dmsecs = None

    for topic, msg, t in bagfile.read_messages(topics=topics):
        if init_time is None:
            init_time = t
        time_diff = (t - init_time)
        delta_msecs = time_diff.secs + time_diff.nsecs / 1e9

        if(count> CUTOFF):
            break
        else:
            count = count+1
            if plot_progress:
                bar.update(count)

        if topic == '/reskin':
            data = np.array(msg.data) # 5 x (T x y z)

            magns = np.array([
                [data[1], data[2], data[3]],
                [data[5], data[6], data[7]],
                [data[9], data[10], data[11]],
                [data[13], data[14], data[15]],
                [data[17], data[18], data[19]],
            ])

            reskin_deque.append((delta_msecs, magns))
            # if len(reskin_deque) > 250:
                # last_reskin_dmsecs = reskin_deque.popleft()[0]
            
            if len(clf_deque) != 0:
                last_clf_dmsecs = clf_deque[0][0]
                if last_clf_dmsecs < last_reskin_dmsecs:
                    clf_deque.popleft()

        elif topic == '/classifier_output':
            clf_deque.append((delta_msecs, msg.data))

        elif topic == '/webcam_image':
            img = bridge.imgmsg_to_cv2(msg)
            images.append(img)
            ax1.cla()
            ax2.cla()
            ax3.cla()
            ax4.cla()
            ax5.cla()
            ax6.cla()
            ax1.set_ylabel("C")
            ax2.set_ylabel("T")
            ax3.set_ylabel("R")
            ax4.set_ylabel("B")
            ax5.set_ylabel("L")
            ax6.set_ylabel("Pred")
            reskin_dmsecs = np.array([x[0] for x in reskin_deque]) # t x 1
            reskin_magns = np.array([x[1] for x in reskin_deque]) # t x 5 x 3
            
            clf_dmsecs = np.array([x[0] for x in clf_deque])
            clf_preds = np.array([x[1] for x in clf_deque])

            for i in range(5):
                # axes[i].scatter([reskin_dmsecs[0], reskin_dmsecs[-1]], [-1, -1], color='white')
                axes[i].plot(reskin_dmsecs, reskin_magns[:, i, 0], 'r')
                axes[i].plot(reskin_dmsecs, reskin_magns[:, i, 1], 'g')
                axes[i].plot(reskin_dmsecs, reskin_magns[:, i, 2], 'b')
            
            axes[5].set_ylim(-1.5, 3.5)
            axes[5].scatter([reskin_dmsecs[0], reskin_dmsecs[-1]], [-1, -1], color='white')
            if len(clf_preds) != 0:
                axes[5].plot(clf_dmsecs, clf_preds)

            fig.canvas.draw()
            img1 = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
                sep='')
            img1  = img1.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            img1 = cv2.cvtColor(img1,cv2.COLOR_RGB2BGR)
            plot_images.append(img1)
            # store imgs in data structure
            scale = float(img1.shape[1]) / img.shape[1]
            img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
            new_img = np.concatenate((img, img1))
            final_images.append(new_img)
    for i in range(10): # keep last frame for 1 second
        final_images.append(new_img)

    if plot_progress:
        bar.finish()
    return final_images
    

if(__name__ == "__main__"):
    parser = argparse.ArgumentParser(description='Visualize a BagFile')
    parser.add_argument('--name', type=str, required=True)
    args = parser.parse_args()
    bagpath = '/home/sashank/catkin_ws/src/tactilecloth/bagfiles/'+args.name+'.bag'
    bag = rosbag.Bag(bagpath)
    CUTOFF = 100000
    final_images = create_video_array(bag, CUTOFF, False)
    height, width, layers = final_images[1].shape
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video=cv2.VideoWriter('/home/sashank/catkin_ws/src/tactilecloth/videos/'+args.name+'_magnetometers.mp4',fourcc,10,(width,height))

    for j in range(len(final_images)):
        video.write(final_images[j])

    cv2.destroyAllWindows()
    video.release()
    pass