#!/usr/bin/python
import matplotlib
matplotlib.use('Agg') # no visual
import create_label
import visualize_magnetometers
import visualize_contact
import plots
import rosbag
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
# from shutil import copyfile
# import pickle
# from sklearn import neighbors

import cv2
CUTOFF = 100000

# iterate over directories 
# call a function to process a single bag

def get_label_from_path(path):
    if '0cloth' in path:
        return 1
    elif '1cloth' in path:
        return 2
    elif '2cloth' in path:
        return 3
    elif '3cloth' in path:
        return 4
    else:
        raise NotImplementedError

def process_bag_without_pred(root_folder):
    bag = rosbag.Bag(root_folder + '/data.bag')

    final_data = create_label.create_data_no_label(bag)
    np.savetxt(root_folder+"/reskin_data.csv", final_data, delimiter=",")
    
    final_images = visualize_magnetometers.create_video_array(bag, CUTOFF)
    height, width, layers = final_images[1].shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    if not os.path.exists(root_folder + '/videos'):
        os.mkdir(root_folder + '/videos')
    video_path = root_folder +'/videos/magnetometers.avi'
    video = cv2.VideoWriter(video_path,fourcc,10,(width,height))
    for j in range(len(final_images)):
        video.write(final_images[j])
    cv2.destroyAllWindows()
    video.release()

    os.system('ffmpeg -y -i %s -vf "fps=10,scale=320:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 %s/videos/magnetometers.gif'
        % (video_path, root_folder))

def process_bag_with_pred(root_folder):
    bag = rosbag.Bag(root_folder + '/data.bag')
    label = get_label_from_path(root_folder)

    final_data = create_label.create_data_from_norm(bag, label)
    np.savetxt(root_folder+"/reskin_data.csv", final_data, delimiter=",")
    
    visualize_magnetometers.plot_all_data(bag, fname=root_folder + '/classifier_pred.png', has_prediction=True)

    final_images = visualize_magnetometers.create_video_array(bag, CUTOFF, has_prediction=True)
    height, width, layers = final_images[1].shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    if not os.path.exists(root_folder + '/videos'):
        os.mkdir(root_folder + '/videos')
    video_path = root_folder +'/videos/magnetometers.avi'
    video = cv2.VideoWriter(video_path,fourcc,10,(width,height))
    for j in range(len(final_images)):
        video.write(final_images[j])
    cv2.destroyAllWindows()
    video.release()

    os.system('ffmpeg -y -i %s -vf "fps=10,scale=320:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 %s/videos/magnetometers.gif'
        % (video_path, root_folder))
 
    # contact_str = check_frame.mark_frames(video_path)
    # markers_path = root_folder+'/'+args.name+'_markers.csv'
    # file1 = open(markers_path,"w+")
    # file1.write(contact_str)
    # file1.close()
    # contact_arr = np.loadtxt(markers_path, delimiter=",")
    # final_images = visualize_contact.create_video_array(bag, final_data, CUTOFF, 500)
    # height, width, layers = final_images[1].shape
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # video_path = root_folder+'/videos/contact.avi'
    # video=cv2.VideoWriter(video_path,fourcc,10,(width,height))
    # for j in range(len(final_images)):
    #     video.write(final_images[j])
    # cv2.destroyAllWindows()
    # video.release()

if(__name__ == "__main__"):
    parser = argparse.ArgumentParser(description='Create Complete DataSet')
    parser.add_argument('--dir', help="directory containing trials of a single type", 
        type=str, default=None)
    parser.add_argument('--path', help="path to a single trial", 
        type=str, default=None)
    parser.add_argument('--predict', help="run process bag with prediction", 
        action='store_true')
        
    args = parser.parse_args()

    if args.dir is not None:
        for dir in sorted(os.listdir(args.dir)):
            process_bag(args.dir + "/" + dir)
    elif args.path is not None:
        if args.predict:
            process_bag_with_pred(args.path)
        else:
            process_bag_without_pred(args.path)

    # root_folder = '/home/sashank/catkin_ws/src/tactilecloth/classification_data/'+args.name
    # video_folder = root_folder+"/videos"
    # os.mkdir(root_folder)
    # os.mkdir(video_folder)
    # bagpath = '/home/sashank/catkin_ws/src/tactilecloth/bagfiles/'+args.name+'.bag'

    # parser = argparse.ArgumentParser(description='Create Complete DataSet')
    # parser.add_argument('--name', type=str, required=True)
    # args = parser.parse_args()
    # root_folder = '/home/sashank/catkin_ws/src/tactilecloth/classification_data/'+args.name
    # video_folder = root_folder+"/videos"
    # os.mkdir(root_folder)
    # os.mkdir(video_folder)
    # bagpath = '/home/sashank/catkin_ws/src/tactilecloth/bagfiles/'+args.name+'.bag'