# todo -- MODULARIZE THE CODE INTO FUNCTIONS, FOR NOW WE CAN IGNORE

import cv2
import rosbag
import numpy as np
from cv_bridge import CvBridge
import cv2 
import time
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser(description='Visualize a BagFile')
parser.add_argument('--name', type=str, required=True)
args = parser.parse_args()
bagpath = '/home/sashank/catkin_ws/src/tactilecloth/bagfiles/'+args.name+'.bag'
bridge = CvBridge()
bag = rosbag.Bag(bagpath)
images = []
cdata = []
plot_images=[]
fig = plt.figure()
count = 0
final_images=[]
for topic, msg, t in bag.read_messages(topics=['/reskin', '/webcam_image']):
    if(count> 100000):
        break
    else:
        count = count+1
        if(count%100 == 0):
            print(count)
    if topic == '/reskin':
        data = np.array(msg.data) # 5 x (T x y z)
        mag1 = np.sqrt(data[1]**2 + data[2]**2 + data[3]**2)
        mag2 = np.sqrt(data[5]**2 + data[6]**2 + data[7]**2)
        mag3 = np.sqrt(data[9]**2 + data[10]**2 + data[11]**2)
        mag4 = np.sqrt(data[13]**2 + data[14]**2 + data[15]**2)
        mag5 = np.sqrt(data[17]**2 + data[18]**2 + data[19]**2)
        # print(mag1, mag2,mag3)
        if(len(cdata)<250):
            cdata.append([mag1,mag2,mag3,mag4,mag5])
        else:
            dat = cdata.pop(0)
            cdata.append([mag1,mag2,mag3,mag4,mag5])


        # store readings in data structure

    elif topic == '/webcam_image':
        img = bridge.imgmsg_to_cv2(msg)
        images.append(img)
        plt.cla()
        cdatan = np.array(cdata)
        plt.plot(cdatan[:,0], 'r')
        plt.plot(cdatan[:,1], 'g')
        plt.plot(cdatan[:,2], 'b')
        plt.plot(cdatan[:,3], 'k')
        plt.plot(cdatan[:,4], 'o')

        fig.canvas.draw()
        img1 = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
            sep='')
        img1  = img1.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        img1 = cv2.cvtColor(img1,cv2.COLOR_RGB2BGR)
        plot_images.append(img1)
        # store imgs in data structure
        new_img = np.concatenate((img, img1))
        final_images.append(new_img)


height,width,layers=final_images[1].shape
print(height, width, layers)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video=cv2.VideoWriter('/home/sashank/catkin_ws/src/tactilecloth/videos/'+args.name+'_magnitudes.avi',fourcc,10,(width,height))

for j in range(len(images)):
    video.write(final_images[j])

cv2.destroyAllWindows()
video.release()

if(__name__ == "__main__"):
    
    pass