import numpy as np
import rosbag
import progressbar
import numpy as np
from cv_bridge import CvBridge
import cv2 
import time
import matplotlib.pyplot as plt
import os
import rospy
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import neighbors
import sklearn
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix

def get_bags_from_dir(dirs):
    bags=[]
    bagpaths=[]
    for curdir in sorted(os.listdir(dirs)):
        bagpath= dirs+"/"+curdir+"/data.bag"
        bagpaths.append(bagpath)
        bag = rosbag.Bag(bagpath)
        bags.append(bag)
    return bags, bagpaths

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
            plt.plot(np.diff(contact_data), 'r')
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
    bar.finish()
    return final_images

def create_clf_video_array(bagfile, data_arr, CUTOFF, PTS_PLOTTED = 250, verbose = False):
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
            contact_data = current_data[:,-3]
            max_cont = np.max(data_arr[:,-3])
            clf_data = current_data[:,-2]
            plt.cla()
            plt.xlim([0, 550])
            plt.ylim([-0.1, max_cont+1])
            plt.title("Contact Signal")
            plt.plot(np.array(contact_data), 'r', alpha = 0.5)
            plt.plot(np.array(clf_data), 'g', alpha=0.5)
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
    bar.finish()
    return final_images

def create_label_from_clf_command(bagfile, label):
    final_data=[]
    t_start = 0
    t_end = 0
    for topic, msg, t in bagfile.read_messages(topics=['/classifier_commands']):
        if topic == '/classifier_commands':
            data = np.array(msg.data) # 5 x (T x y z)
            if(data=="start"):
                t_start = t
            if(data=="end"):
                t_end = t
#     print(t_start, t_end)
    for topic, msg, t in bagfile.read_messages(topics=['/reskin']):
        if topic == '/reskin':
            data = np.array(msg.data) # 5 x (T x y z)
            new_data = [data[1], data[2], data[3], data[5], data[6], data[7],
            data[9], data[10], data[11], data[13], data[14], data[15], data[17], data[18], data[19]]
            dur = t - t_start
            # print(dur.secs + dur.nsecs / 1e9)
            if(dur.secs + dur.nsecs / 1e9 >= 0.2 and t<t_end):
#                 print("inside")
                new_data.append(label)
            else:
                new_data.append(0)
            new_data.append(int(str(t)))
            final_data.append(new_data)
    final_data = np.array(final_data)
    return final_data

def get_labelled_training_data(bags_ls, bagpaths_ls, num_trials = 35):
    X=[]
    Y=[]
    for i in range(len(bags_ls)):
        for j in range(num_trials):
            bf = bags_ls[i][j]
            bp = bagpaths_ls[i][j]
            if("0cloth" in bp):
                labelled_data = create_label_from_clf_command(bf,  1)
            if("1cloth" in bp):
                labelled_data = create_label_from_clf_command(bf,  2)
            if("2cloth" in bp):
                labelled_data = create_label_from_clf_command(bf,  3)
            X.append(labelled_data[:,:-2])
            Y.append(labelled_data[:,-2])
    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)
    return X,Y

def get_labelled_testing_data(bags_ls, bagpaths_ls, num_tests = 10):
    X=[]
    Y=[]
    for i in range(len(bags_ls)):
        for j in range(1, num_tests):
            bf = bags_ls[i][-j]
            bp = bagpaths_ls[i][-j]
            if("0cloth" in bp):
                labelled_data = create_label_from_clf_command(bf,  1)
            if("1cloth" in bp):
                labelled_data = create_label_from_clf_command(bf,  2)
            if("2cloth" in bp):
                labelled_data = create_label_from_clf_command(bf,  3)
            X.append(labelled_data[:,:-2])
            Y.append(labelled_data[:,-2])
    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)
    return X,Y

def train_classifier(X,Y, nn=10):
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)
    clf = neighbors.KNeighborsClassifier(nn, weights="distance")
    clf.fit(X, Y.ravel())
    return clf, scaler

def test_classifier(clf, scaler, x, y):
    x = scaler.transform(x)
    y_pred = clf.predict(x)
    score = balanced_accuracy_score(y, y_pred)
    print("balanced accuracy: ",score)
    best_cf = confusion_matrix(y, y_pred)
    print("confusion_matrix: ", best_cf)
    return y_pred

def save_labelled_data(all_bag_files, all_bag_paths):
    for i in range(len(all_bag_files)):
        for j in range(len(all_bag_files[0])):
            bf = all_bag_files[i][-j]
            bp = all_bag_paths[i][-j]
            if("0cloth" in bp):
                labelled_data = create_label_from_clf_command(bf,  1)
            if("1cloth" in bp):
                labelled_data = create_label_from_clf_command(bf,  2)
            if("2cloth" in bp):
                labelled_data = create_label_from_clf_command(bf,  3)
            csv_path = bp.replace("data.bag","reskin_labelled.csv")
            np.savetxt(csv_path, labelled_data, delimiter=",")
    return None

if(__name__ == "__main__"):
    p1 = "/media/tweng/ExtraDrive2/fabric_touch/bagfiles/franka_norub_folded_random_18feb/0cloth_norub_auto_0"
    p2 = "/media/tweng/ExtraDrive2/fabric_touch/bagfiles/franka_norub_folded_random_18feb/1cloth_norub_auto_0"
    p3 = "/media/tweng/ExtraDrive2/fabric_touch/bagfiles/franka_norub_folded_random_18feb/2cloth_norub_auto_0"

    cloth0_bags, c0bps = get_bags_from_dir(p1)
    cloth1_bags, c1bps = get_bags_from_dir(p2)
    cloth2_bags, c2bps = get_bags_from_dir(p3)
    # ind = 20
    # final_data = create_label_from_clf_command(cloth2_bags[ind], 1)
    # final_images = create_video_array(cloth2_bags[ind], final_data, 100000, PTS_PLOTTED = 250, verbose = False)
    # height, width, layers = final_images[1].shape
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # video_path = "dbg.avi"
    # video = cv2.VideoWriter(video_path,fourcc,10,(width,height))
    # for j in range(len(final_images)):
    #     video.write(final_images[j])
    # cv2.destroyAllWindows()
    # video.release()

    all_bag_files = [cloth0_bags,cloth1_bags,cloth2_bags]
    all_bag_paths = [ c0bps , c1bps , c2bps ]
    save_labelled_data(all_bag_files, all_bag_paths)
    # X,Y =  get_labelled_training_data(all_bag_files, all_bag_paths, num_trials = 35)
    # clf, scalar = train_classifier(X,Y)
    # x,y = get_labelled_testing_data(all_bag_files, all_bag_paths, num_tests = 10)
    # y_pred = test_classifier(clf, scalar, x, y)

    # print(x.shape, y.shape, y_pred.shape)

    # data_arr = np.hstack([x,y.reshape(-1,1),y_pred.reshape(-1,1)])
    # np.save("fn_data.npy", data_arr)
    # ind = 44
    # final_data = create_label_from_clf_command(cloth2_bags[ind], 3)
    # y_pred = test_classifier(clf, scalar, final_data[:,:-2], final_data[:,-2])
    # # print(y_pred, final_data[:,-2])
    # data_arr=np.hstack([final_data[:,:-1], y_pred.reshape(-1,1), final_data[:,-1].reshape(-1,1)])

    # final_images =create_clf_video_array(cloth2_bags[ind], data_arr, 100000, PTS_PLOTTED = 250, verbose = False)
    # height, width, layers = final_images[1].shape
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # video_path = "dbg_clf_"+str(ind)+".avi"
    # video = cv2.VideoWriter(video_path,fourcc,10,(width,height))
    # for j in range(len(final_images)):
    #     video.write(final_images[j])
    # cv2.destroyAllWindows()
    # video.release()








