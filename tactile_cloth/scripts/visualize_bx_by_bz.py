import cv2
import rosbag
import numpy as np
from cv_bridge import CvBridge
import cv2 
import time
import matplotlib.pyplot as plt
import argparse
def create_video_array(bagpath, CUTOFF, verbose = False):
    bridge = CvBridge()
    bag = rosbag.Bag(bagpath)
    images = []
    bx = []
    by = []
    bz = []
    ct = []
    plot_images=[]
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True)
    axes = [ax1, ax2, ax3, ax4]
    fig.suptitle("Magnetometer Readings: Bx, By, Bz")
    count = 0
    final_images=[]
    for topic, msg, t in bag.read_messages(topics=['/reskin', '/webcam_image', 'contact_type']):
        if(count> CUTOFF):
            break
        else:
            count = count+1
            if(count%100 == 0):
                if(verbose):
                    print(count)
                pass
        if topic == '/reskin':
            data = np.array(msg.data) # 5 x (T x y z)
            if(len(bx)<250):
                bx.append([data[1], data[5], data[9], data[13], data[17]])
            else:
                dat = bx.pop(0)
                bx.append([data[1], data[5], data[9], data[13], data[17]])
            
            if(len(by)<250):
                by.append([data[2], data[6], data[10], data[14], data[18]])
            else:
                dat = by.pop(0)
                by.append([data[2], data[6], data[10], data[14], data[18]])
            
            if(len(bz)<250):
                bz.append([data[3], data[7], data[11], data[15], data[19]])
            else:
                dat = bz.pop(0)
                bz.append([data[3], data[7], data[11], data[15], data[19]])
            
        elif(topic == '/contact_type'):
            # print("entered")
            data = msg.data
            print(data)
            if(len(ct)<250):
                ct.append(data)
            else:
                dat = ct.pop(0)
                ct.append(data)


        elif topic == '/webcam_image':
            # print(count)
            img = bridge.imgmsg_to_cv2(msg)
            images.append(img)
            img = images[-1]
            ax1.cla()
            ax2.cla()
            ax3.cla()
            ax4.cla()

            bx_n = np.array(bx)
            by_n = np.array(by)
            bz_n = np.array(bz)

            ax1.plot(bx_n[:,0],'r')
            ax1.plot(bx_n[:,1],'g')
            ax1.plot(bx_n[:,2],'b')
            ax1.plot(bx_n[:,3],'k')
            ax1.plot(bx_n[:,4],'o')

            ax2.plot(by_n[:,0], 'r')
            ax2.plot(by_n[:,1], 'g')
            ax2.plot(by_n[:,2], 'b')
            ax2.plot(by_n[:,3], 'k')
            ax2.plot(by_n[:,4], 'o')

            ax3.plot(bz_n[:,0], 'r')
            ax3.plot(bz_n[:,1], 'g')
            ax3.plot(bz_n[:,2], 'b')
            ax3.plot(bz_n[:,3], 'k')
            ax3.plot(bz_n[:,4], 'o')

            ax4.plot(ct[:], 'r')

            fig.canvas.draw()
            img1 = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
                sep='')
            img1  = img1.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            img1 = cv2.cvtColor(img1,cv2.COLOR_RGB2BGR)
            plot_images.append(img1)
            # store imgs in data structure
            new_img = np.concatenate((img, img1))
            final_images.append(new_img)

    return final_images
    height,width,layers=final_images[1].shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video=cv2.VideoWriter('/home/sashank/catkin_ws/src/tactilecloth/videos/'+args.name+'_bx_by_bz.avi',fourcc,10,(width,height))

    for j in range(len(images)):
        video.write(final_images[j])

    cv2.destroyAllWindows()
    video.release()


if(__name__ == "__main__"):
    parser = argparse.ArgumentParser(description='Visualize a BagFile')
    parser.add_argument('--name', type=str, required=True)
    args = parser.parse_args()
    bagpath = '/home/sashank/catkin_ws/src/tactilecloth/bagfiles/'+args.name+'.bag'
    CUTOFF = 10000
    finale_video = create_video_array(bagpath, CUTOFF, True)
    height,width,layers=finale_video[1].shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video=cv2.VideoWriter('/home/sashank/catkin_ws/src/tactilecloth/videos/'+args.name+'_bx_by_bz.avi',fourcc,10,(width,height))

    for j in range(len(finale_video)):
        video.write(finale_video[j])

    cv2.destroyAllWindows()
    video.release()
    
    pass