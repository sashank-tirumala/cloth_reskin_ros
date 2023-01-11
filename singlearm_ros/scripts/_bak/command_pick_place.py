#!/usr/bin/env python
import os
import cv2
import rospy
import actionlib
import message_filters
import numpy as np
np.set_printoptions(suppress=True)

from copy import deepcopy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Pose, Quaternion

from utils.marker_visualizer import MarkerVisualizer
from singlearm.msg import FoldAction, FoldGoal, ResetAction, ResetGoal
# FoldDualAction, FoldDualGoal

import tf
from tf.transformations import *

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from utils.model_load_util import load_model_txt
from utils.projection import project_to_3D
from models.flownet import FlowNetSmall
from models.picknet import FlowPickSplit
from models.raft_core.raft import RAFT

import argparse

# CLOTH = 'real_high' # 'real_lblue' # 'real_shirt' # 'real_rect' 

def ccw(A,B,C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

class PickPlaceServer(object):
    def __init__(self, method):
        self.D = np.array(rospy.get_param('D'))
        self.K = np.array(rospy.get_param('K'))
        self.K = np.reshape(self.K, (3, 3))

        self.marker_viz = MarkerVisualizer()

        self.bridge = CvBridge()
        self.depth_im = None
        self.rgb_im = None
        self.depthsub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
        self.rgbsub = message_filters.Subscriber('/camera/color/image_rect_color', Image)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.depthsub, self.rgbsub], 10, 0.1)
        self.ts.registerCallback(self.cb)

        rospy.loginfo("Waiting for reset action server...")
        self.reset_client = actionlib.SimpleActionClient('reset', ResetAction)
        self.reset_client.wait_for_server()
        rospy.loginfo("Connected to reset action server.")

        # rospy.loginfo("Waiting for dual fold action server...")
        # self.dual_fold_client = actionlib.SimpleActionClient('dualfold', FoldDualAction)
        # self.dual_fold_client.wait_for_server()
        # rospy.loginfo("Connected to dual fold action server.")

        rospy.loginfo("Waiting for single fold action server...")
        self.single_fold_client = actionlib.SimpleActionClient('singlefold', FoldAction)
        self.single_fold_client.wait_for_server()
        rospy.loginfo("Connected to single fold action server.")

        self.sub1 = rospy.Subscriber('fold', String, self.run)
        self.sub2 = rospy.Subscriber('reset', String, self.reset)

        self.listener = tf.TransformListener()

        self.method = method

    def cb(self, depth_msg, rgb_msg):
        depth_im = self.bridge.imgmsg_to_cv2(depth_msg) / 1000.0
        rgb_im = self.bridge.imgmsg_to_cv2(rgb_msg)[:, 80:-80]
        rgb_im = cv2.resize(rgb_im, (200, 200))
        # rgb_im = cv2.cvtColor(rgb_im, cv2.COLOR_BGR2RGB)
        self.depth_im = np.nan_to_num(depth_im)
        self.rgb_im = rgb_im

    def create_marker(self, x, y, z, colors=[1, 0, 0, 1], marker_id=0):
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        pose.orientation.w = 1.0
        self.marker_viz.set_marker(pose, colors=colors, marker_id=marker_id, frame='base', marker_type=2)

    def create_pose(self, x, y, z):
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z # + 0.2
        pose.orientation.x = 0.000000
        pose.orientation.y = -1.000000
        pose.orientation.z = 0.000000
        pose.orientation.w = 0.000000
        return pose

    def reset(self, msg):
        goal = ResetGoal()
        self.reset_client.send_goal(goal)
        self.reset_client.wait_for_result()
        response = self.reset_client.get_result()
        rospy.loginfo(response)

    def save_final_images(self, savefig_path):
        final_depth = deepcopy(self.depth_im)
        final_rgb = deepcopy(self.rgb_im)
        final_rgb = cv2.cvtColor(final_rgb, cv2.COLOR_BGR2RGB)
        savepath = savefig_path.replace('.png', '_final.npy')
        np.save(savepath, final_depth)
        savergb_path = savefig_path.replace('.png', '_final.png')
        cv2.imwrite(savergb_path, final_rgb)
        rospy.loginfo("Saved final images.")

    def run(self, msg, thresh=30):
        rospy.loginfo("Received fold request...")
        rgb_obs = deepcopy(self.rgb_im)
        depth_obs = deepcopy(self.depth_im)
        test_dir, goal_name = msg.data.split('-')
        # pick1, pick2, place1, place2, info = 
        pick1, place1, info = self.method.run(rgb_obs, depth_obs, goal_name=goal_name, test_dir=test_dir)
        
        # Record video
        savefig_path = info['savefig_path']
        
        # savevid_path = savefig_path.replace('png', 'mp4')

        # inp = raw_input("Plan? [y/n]: ")
        # if inp != 'y':
        #     return 

        # dist = min(np.linalg.norm(pick1 - pick2), np.linalg.norm(place1 - place2))
        # if dist > thresh:
        #     rospy.loginfo("Executing dual arm fold (dist: {})".format(dist))
        #     figpath_dual = savefig_path.replace('.png', '_dual.png')
        #     vidpath_dual = figpath_dual.replace('png', 'mp4')
        #     self.execute_dual_arm_fold(vidpath_dual, depth_obs, pick1, place1, pick2, place2)
        #     self.save_final_images(figpath_dual)
        #     inp = raw_input("Try single arm? y/n: ")
        #     if inp == 'y':
        #         rospy.loginfo("Executing single arm fold (dist: {})".format(dist))
        #         figpath_single = savefig_path.replace('.png', '_single.png')
        #         vidpath_single = figpath_single.replace('png', 'mp4')
        #         self.execute_single_arm_fold(vidpath_single, depth_obs, pick1, place1)
        #         self.save_final_images(figpath_single)
        # else:
        rospy.loginfo("Executing single arm fold (dist: {})".format(dist))
        figpath_single = savefig_path.replace('.png', '_single.png')
        vidpath_single = figpath_single.replace('png', 'mp4')
        self.execute_single_arm_fold(vidpath_single, depth_obs, pick1, place1)
        self.save_final_images(figpath_single)

    def execute_single_arm_fold(self, savevid_path, depth_im, pick1_uv, place1_uv):
        # Get world to camera pose
        # self.listener.waitForTransform("base", "camera_color_optical_frame", rospy.Time(0), rospy.Duration(3.0))
        (trans,rot) = self.listener.lookupTransform('base', "camera_color_optical_frame", rospy.Time(0))

        pick1 = int(pick1_uv[0]/199.0*479), int(pick1_uv[1]/199.0*479+80)
        place1 = int(place1_uv[0]/199.0*479), int(place1_uv[1]/199.0*479+80)

        x, y, z = project_to_3D(pick1, depth_im, trans, rot, self.D, self.K)
        self.create_marker(x, y, z, colors=[1, 0, 0, 1], marker_id=0)
        pick1_pose = self.create_pose(x, y, z)

        x, y, z = project_to_3D(place1, depth_im, trans, rot, self.D, self.K)
        self.create_marker(x, y, z, colors=[0, 1, 0, 1], marker_id=2)
        place1_pose = self.create_pose(x, y, z)

        # rospy.loginfo(pick1_pose)   
        # rospy.loginfo(place1_pose)

        s = String()
        s.data = savevid_path
        goal = FoldGoal(pick1_pose, place1_pose, s)
        self.single_fold_client.send_goal(goal)
        self.single_fold_client.wait_for_result()
        response = self.single_fold_client.get_result()
        return response

    # def execute_dual_arm_fold(self, savevid_path, depth_im, pick1_uv, place1_uv, pick2_uv, place2_uv, thresh=30):
    #     # Get world to camera pose
    #     # self.listener.waitForTransform("base", "camera_color_optical_frame", rospy.Time(0), rospy.Duration(3.0))
    #     (trans,rot) = self.listener.lookupTransform('base', "camera_color_optical_frame", rospy.Time(0))

    #     # pick1 = int(pick1_uv[0]/199.0*309+90), int(pick1_uv[1]/199.0*309+150)
    #     # place1 = int(place1_uv[0]/199.0*309+90), int(place1_uv[1]/199.0*309+150)
    #     # pick2 = int(pick2_uv[0]/199.0*309+90), int(pick2_uv[1]/199.0*309+150)
    #     # place2 = int(place2_uv[0]/199.0*309+90), int(place2_uv[1]/199.0*309+150)
    #     pick1 = int(pick1_uv[0]/199.0*479), int(pick1_uv[1]/199.0*479+80)
    #     place1 = int(place1_uv[0]/199.0*479), int(place1_uv[1]/199.0*479+80)
    #     pick2 = int(pick2_uv[0]/199.0*479), int(pick2_uv[1]/199.0*479+80)
    #     place2 = int(place2_uv[0]/199.0*479), int(place2_uv[1]/199.0*479+80)

    #     x, y, z = project_to_3D(pick1, depth_im, trans, rot, self.D, self.K)
    #     self.create_marker(x, y, z, colors=[1, 0, 0, 1], marker_id=0)
    #     pick1_pose = self.create_pose(x, y, z)

    #     x, y, z = project_to_3D(pick2, depth_im, trans, rot, self.D, self.K)
    #     self.create_marker(x, y, z, colors=[1, 0, 0, 1], marker_id=1)
    #     pick2_pose = self.create_pose(x, y, z)

    #     x, y, z = project_to_3D(place1, depth_im, trans, rot, self.D, self.K)
    #     self.create_marker(x, y, z, colors=[0, 1, 0, 1], marker_id=2)
    #     place1_pose = self.create_pose(x, y, z)

    #     x, y, z = project_to_3D(place2, depth_im, trans, rot, self.D, self.K)
    #     self.create_marker(x, y, z, colors=[0, 1, 0, 1], marker_id=3)
    #     place2_pose = self.create_pose(x, y, z)

    #     # rospy.loginfo(pick1_pose)
    #     # rospy.loginfo(pick2_pose)
    #     # rospy.loginfo(place1_pose)
    #     # rospy.loginfo(place2_pose)

    #     s = String()
    #     s.data = savevid_path
    #     goal = FoldDualGoal(pick1_pose, place1_pose, pick2_pose, place2_pose, s)
    #     self.dual_fold_client.send_goal(goal)
    #     self.dual_fold_client.wait_for_result()
    #     response = self.dual_fold_client.get_result()

    #     return response
        
    def spin(self):
        while not rospy.is_shutdown():
            self.marker_viz.publish()

class ManualCommand(object):
    def __init__(self):
        self.pick1_uv = []
        self.pick2_uv = []
        self.place1_uv = []
        self.place2_uv = []

    def winclicked(self, event, v, u, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            rospy.loginfo('Selected grasp point u: %d v: %d' % (u, v))
            if param == 'pick1':
                self.pick1_uv = [u, v]
            elif param == 'pick2':
                self.pick2_uv = [u, v]
            elif param == 'place1':
                self.place1_uv = [u, v]
            elif param == 'place2':
                self.place2_uv = [u, v]

    def create_window(self, rgb_im, wintitle, selecttype):
        cv2.namedWindow(wintitle)
        cv2.setMouseCallback(wintitle, self.winclicked, selecttype)
        cv2.imshow(wintitle, rgb_im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def run(self, rgb_obs, depth_obs, rgb_goal, depth_goal):
        depth_im = deepcopy(self.depth_im)
        rgb_im = deepcopy(self.rgb_im)
        
        self.create_window(rgb_im, 'Click first pick point and press q', 'pick1')
        self.create_window(rgb_im, 'Click first place point and press q', 'place1')
        self.create_window(rgb_im, 'Click second pick point and press q', 'pick2')
        self.create_window(rgb_im, 'Click second place point and press q', 'place2')
        # Debug
        # self.pick1_uv = [350, 200]
        # self.place1_uv = [100, 200]
        # self.pick2_uv =  [350, 500]
        # self.place2_uv =  [100, 500]

        pick1_uv, place1_uv, pick2_uv, place2_uv = self.pick1_uv, self.place1_uv, self.pick2_uv, self.place2_uv

        self.pick1_uv = []
        self.place1_uv = []
        self.pick2_uv = []
        self.place2_uv = []
        return pick1_uv, place1_uv, pick2_uv, place2_uv

class FabricFlowNet(object):
    def __init__(self, flow_wt, args):
        self.args = args

        if not self.args.noflow:
            # flow model
            rospy.loginfo("Loading Flownet...")
            self.flow_wt = flow_wt
            flow_path = '/home/tweng/ws1/src/bimanual_folding/bimanual_ros/scripts/models/weights/{}.txt'.format(flow_wt)
            self.flow = FlowNetSmall(input_channels=2).cuda()
            load_model_txt(self.flow, flow_path)
            self.flow.eval()

        # raft model
        # # rospy.loginfo("Loading RAFT...")
        # raftwts = '595000_raft-towel-spatialaug-0.65-flip'
        # raft_path = '/home/tweng/ws1/src/bimanual_folding/bimanual_ros/scripts/models/weights/{}.txt'.format(raftwts)
        # args = argparse.Namespace(small=False, 
        #                           mixed_precision=True, 
        #                           alternate_corr=False, 
        #                           iters=12, 
        #                           gpus=[0],
        # )
        #                         #   image_size=(200, 200))
        # self.flow = torch.nn.DataParallel(RAFT(args))
        # load_model_txt(self.flow, raft_path)
        # # from functools import partial
        # # import pickle
        # # pickle.load = partial(pickle.load, encoding="latin1")
        # # pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
        # # self.flow.load_state_dict(torch.load(raft_path, map_location=lambda storage, loc: storage, pickle_module=pickle))
        # # self.flow.load_state_dict(torch.load(raft_path))
        # self.flow.cuda()
        # self.flow.eval()

        # pick model
        # self.pick_wt = pick_wt
        weight_path = '/home/tweng/ws1/src/bimanual_folding/bimanual_ros/scripts/models/weights/{}.txt'
        # first_path = weight_path.format('first_{}'.format(pick_wt))
        first_path = weight_path.format(self.args.pickwt1)
        self.first = FlowPickSplit(2, 200).cuda()
        load_model_txt(self.first, first_path)
        self.first.eval()

        # second_path = weight_path.format('second_{}'.format(pick_wt))
        second_path = weight_path.format(self.args.pickwt2)
        self.second = FlowPickSplit(3, 200, second=True).cuda()
        load_model_txt(self.second, second_path)
        self.second.eval()

        if self.args.noflow:
            self.place1 = FlowPickSplit(2, 200).cuda()
            first_path = weight_path.format(self.args.placewt1)
            load_model_txt(self.place1, first_path)
            # self.place1.load_state_dict(torch.load(first_path))
            self.place1.eval()

            self.place2 = FlowPickSplit(3, 200, second=True).cuda()
            second_path = weight_path.format(self.args.placewt2)
            load_model_txt(self.place2, second_path)
            # self.place2.load_state_dict(torch.load(second_path))
            self.place2.eval()

        self.cfgs = {
            'prob_type': 'sigmoid',
            'im_width': 200
        }

    def get_pt(self, logits, min_r=3):
        # select 2 pts with NMS
        N = logits.size(0)
        W = logits.size(2)

        if self.cfgs['prob_type'] == 'sigmoid':
            probs = torch.sigmoid(logits)
            probs = probs.view(N,1,W*W)
        else:
            probs = F.softmax(logits.flatten(-2), -1)

        val,idx = torch.max(probs[:,0], 1)
        u = (idx // 20) * 10
        v = (idx % 20) * 10

        return u.item(),v.item()

    def get_gaussian(self, u, v, sigma=5, size=None):
        if size is None:
            size = self.cfgs["im_width"]

        x0, y0 = torch.Tensor([u]).cuda(), torch.Tensor([v]).cuda()
        x0 = x0[:, None]
        y0 = y0[:, None]

        N = 1
        num = torch.arange(size).float()
        # x, y = torch.vstack([num]*N).cuda(), torch.vstack([num]*N).cuda()
        x, y = torch.stack([num]*N, dim=0).cuda(), torch.stack([num]*N, dim=0).cuda()
        gx = torch.exp(-(x-x0)**2/(2*sigma**2))
        gy = torch.exp(-(y-y0)**2/(2*sigma**2))
        g = torch.einsum('ni,no->nio', gx, gy)

        gmin = g.flatten().reshape(g.shape[0], -1).min(dim=1)[0]
        gmax = g.flatten().reshape(g.shape[0], -1).max(dim=1)[0]
        # gmin = g.amin(dim=(1,2))
        # gmax = g.amax(dim=(1,2))
        g = (g - gmin[:,None,None])/(gmax[:,None,None] - gmin[:,None,None])
        g = g.unsqueeze(1)

        if False:
            import matplotlib.pyplot as plt
            for i in range(g.shape[0]):
                plt.imshow(g[i].squeeze().detach().cpu().numpy())
                plt.show()

        return g

    def get_flow_place_pt(self, u,v, flow):
        flow_u_idxs = np.argwhere(flow[0,:,:])
        flow_v_idxs = np.argwhere(flow[1,:,:])
        nearest_u_idx = flow_u_idxs[((flow_u_idxs - [u,v])**2).sum(1).argmin()]
        nearest_v_idx = flow_v_idxs[((flow_v_idxs - [u,v])**2).sum(1).argmin()]

        flow_u = flow[0,nearest_u_idx[0],nearest_u_idx[1]]
        flow_v = flow[1,nearest_v_idx[0],nearest_v_idx[1]]

        new_u = u + flow_u
        new_v = v + flow_v

        return new_u,new_v

    # def load_real_depth(self, path, rgb):
    def preprocess(self, depth, rgb, cropdims=[5, -5, 85, -85]):
        """
        Load depth collected at 55 cm above the table 
        Match depth to sim image
        """
        if depth.shape[-1] == 3:
            depth = depth[:, :, 0]

        if depth.dtype == np.uint8: # test goal
            depth = depth / 255.0
        
        precropped = depth.shape[0] == 200
        if not precropped: # 480 x 640
            x1, x2, y1, y2 = cropdims
            # rgb = cv2.resize(rgb[x1:x2, y1:y2], (200, 200))
            # rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            depth = depth[x1:x2, y1:y2].astype(np.float32) # 310 x 310
            mask = (depth <= 0).astype(np.uint8)*255
            depth = cv2.inpaint(depth, mask, 3, cv2.INPAINT_NS)
            depth = cv2.resize(depth, (200, 200))
        
        # Adjust real table height to sim
        # sim_table_height = 0.45
        sim_table_height = 0.65
        # real_table_height_avg = 0.48766267
        real_table_height_avg = 0.6954650656405612
        table_height_diff = real_table_height_avg - sim_table_height
        depth -= table_height_diff

        if not precropped: # 480 x 640
            # Get background mask

            hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
            if self.args.clothtype == 'real_high': # towel
                cloth_lower = np.array([0,60,0],np.uint8)
                cloth_upper = np.array([24,255,93],np.uint8)
            elif self.args.clothtype == 'real_rect':
                cloth_lower = np.array([0,90,0],np.uint8)
                cloth_upper = np.array([20,255,255],np.uint8)
            elif self.args.clothtype == 'real_shirt':
                cloth_lower = np.array([0,50,0],np.uint8)
                cloth_upper = np.array([20,255,255],np.uint8)
            elif self.args.clothtype == 'real_lblue':
                cloth_lower = np.array([0,150,0],np.uint8)
                cloth_upper = np.array([25,255,255],np.uint8)

            cloth_mask = cv2.inRange(hsv, cloth_lower, cloth_upper)
            kernel = np.ones((9,9),np.uint8)
            cloth_mask = cv2.morphologyEx(cloth_mask, cv2.MORPH_CLOSE, kernel)
            mask = deepcopy(cloth_mask)
            mask[cloth_mask == 0] = 255
            mask[cloth_mask == 255] = 0

            # Apply bg mask to depth
            depth[mask > 0] = 0
        
        return depth

    def nearest_to_mask(self, u, v, depth):
        mask_idx = np.argwhere(depth)
        nearest_idx = mask_idx[((mask_idx - [u,v])**2).sum(1).argmin()]

        return nearest_idx

    def run(self, rgb_obs, depth_obs, goal_name, test_dir='real', preprocess=True, dbg_info={}):
        if dbg_info != {}:
            rgb_goal = dbg_info['rgb_goal']
            depth_goal = dbg_info['depth_goal']
        else:
            # Preprocess images
            rgb_goal = cv2.imread("/home/tweng/ws1/src/bimanual_folding/bimanual_ros/scripts/goals/real_high/{}.png".format(goal_name))
            rgb_goal = cv2.imread("/home/tweng/ws1/src/bimanual_folding/bimanual_ros/scripts/goals/{}/{}.png".format(self.args.clothtype, goal_name))
            depth_goal = np.load("/home/tweng/ws1/src/bimanual_folding/bimanual_ros/scripts/goals/real_high/{}.npy".format(goal_name))[0, :, :] / 1000.0
            depth_goal = np.load("/home/tweng/ws1/src/bimanual_folding/bimanual_ros/scripts/goals/{}/{}.npy".format(self.args.clothtype, goal_name))[0, :, :] / 1000.0
            
            # Sanity check
            if self.args.debug:
                rgb_obs = cv2.imread("/home/tweng/ws1/src/bimanual_folding/bimanual_ros/scripts/goals/sim_high/open_2side_high.png")
                rgb_obs = cv2.cvtColor(rgb_obs, cv2.COLOR_RGB2BGR)
                depth_obs = cv2.imread("/home/tweng/ws1/src/bimanual_folding/bimanual_ros/scripts/goals/sim_high/open_2side_high_depth.png")[:, :, 0] / 255.0
                depth_obs = np.nan_to_num(depth_obs)
                rgb_goal = cv2.imread("/home/tweng/ws1/src/bimanual_folding/bimanual_ros/scripts/goals/sim_high/one_corn_in_0_high.png")
                depth_goal = cv2.imread("/home/tweng/ws1/src/bimanual_folding/bimanual_ros/scripts/goals/sim_high/one_corn_in_0_high_depth.png")[:, :, 0] / 255.0

            rgb_goal = cv2.cvtColor(rgb_goal, cv2.COLOR_RGB2BGR)
            depth_goal = np.nan_to_num(depth_goal)

        if not self.args.debug and preprocess:
            depth_obs = self.preprocess(depth_obs, rgb_obs, cropdims=[90, -80, 150, -180])
            depth_goal = self.preprocess(depth_goal, rgb_goal, cropdims=[90, -80, 150, -180])
        mask = depth_obs == 0

        with torch.no_grad():
            # # inp = np.stack([depth_obs/255.0, depth_goal/255.0], axis=2)
            # if dbg_info != {} and dbg_info['flow_type'] == 'sim':
            #     inp = np.stack([sim_depth_obs, sim_depth_goal], axis=2)
            #     inp_t = torch.from_numpy(inp).float().permute(2, 0, 1).unsqueeze(0).cuda()
            #     flow = self.flow(inp_t) # [1, 2, 200, 200]
            #     flow[:, :, sim_mask] = 0
            # else:

            inp = np.stack([depth_obs, depth_goal], axis=2)
            obs = torch.from_numpy(depth_obs).float().unsqueeze(0).unsqueeze(0).cuda()
            nobs = torch.from_numpy(depth_goal).float().unsqueeze(0).unsqueeze(0).cuda()
            inp_t = torch.from_numpy(inp).float().permute(2, 0, 1).unsqueeze(0).cuda()

            if self.args.noflow:
                x1 = inp_t
            else:
                flow = self.flow(inp_t) # [1, 2, 200, 200]
                flow[:, :, mask] = 0
                x1 = flow

            # im_o = torch.from_numpy(depth_obs).float().unsqueeze(0).unsqueeze(0).cuda()
            # im_g = torch.from_numpy(depth_goal).float().unsqueeze(0).unsqueeze(0).cuda()
            # _, flow = self.flow.module(im_o, im_g, test_mode=True) # [1, 2, 200, 200]
            # flow[:, :, mask] = 0

            # if dbg_info != {} and dbg_info['pick_type'] == 'sim':
            #     im_o = torch.from_numpy(sim_depth_obs).float().unsqueeze(0).unsqueeze(0).cuda()
            # else:
            # im_o = torch.from_numpy(depth_obs).float().unsqueeze(0).unsqueeze(0).cuda()
            # x1 = torch.cat([im_o, flow], dim=1)
            # x1 = flow
            logits1 = self.first(x1)
            pick_u1, pick_v1 = self.get_pt(logits1)
            pick1_gau = self.get_gaussian(pick_u1,pick_v1)
            
            # x2 = torch.cat([im_o, flow, pick1_gau], dim=1)
            if self.args.noflow:
                x2 = torch.cat([obs, nobs, pick1_gau], dim=1)
            else:
                x2 = torch.cat([flow, pick1_gau], dim=1)
            logits2 = self.second(x2)
            pick_u2, pick_v2 = self.get_pt(logits2)
            
            pred1 = np.array([pick_u1, pick_v1])
            pred2 = np.array([pick_u2, pick_v2])

            pick1 = self.nearest_to_mask(pick_u1, pick_v1, depth_obs)
            pick2 = self.nearest_to_mask(pick_u2, pick_v2, depth_obs)

            pickmask_u1,pickmask_v1 = pick1
            pickmask_u2,pickmask_v2 = pick2
            if self.args.noflow:
                logits1pl = self.place1(x1)
                place_u1,place_v1 = self.get_pt(logits1pl)
                place1_gau = self.get_gaussian(place_u1,place_v1)

                x2pl = torch.cat([obs, nobs, place1_gau], dim=1)

                logits2pl = self.place2(x2pl)
                place_u2,place_v2 = self.get_pt(logits2pl)

                # swap if intersecting
                if intersect((pickmask_u1,pickmask_v1),(place_u1,place_v1),(pickmask_u2,pickmask_v2),(place_u2,place_v2)):
                    place1 = np.array([place_u2, place_v2])
                    place2 = np.array([place_u1, place_v1])
                else:
                    place1 = np.array([place_u1, place_v1])
                    place2 = np.array([place_u2, place_v2])
            else:
                flow_arr = flow.detach().cpu().numpy()[0]
                place_u1, place_v1 = self.get_flow_place_pt(pickmask_u1,pickmask_v1,flow_arr)
                place_u2, place_v2 = self.get_flow_place_pt(pickmask_u2,pickmask_v2,flow_arr)
                place1 = np.array([place_u1, place_v1])
                place2 = np.array([place_u2, place_v2])

        if True: # visualize
            im1 = depth_obs
            im2 = depth_goal
            fig, ax = plt.subplots(1, 3, figsize=(32, 16))
            # if dbg_info != {}:
                # plt.title("Flow weight: {}\nPick wt: {}\nFlow type: {}, Pick type: {}".format(self.flow_wt, self.pick_wt, dbg_info['flow_type'], dbg_info['pick_type']))
            ax[0].imshow(rgb_obs)
            ax[0].imshow(im1, vmin=0.6, vmax=0.7, alpha=0.5)
            s = 100
            ax[0].scatter([pick_v1], [pick_u1], facecolors='none', edgecolors='red', marker='o', s=s, label='pred 1', alpha=0.7)
            ax[0].scatter([pick_v2], [pick_u2], facecolors='none', edgecolors='orange', marker='o', s=s, label='pred 2', alpha=0.7)
            ax[0].scatter([pick1[1]], [pick1[0]], color='red', marker='+', s=s, label='pick 1', alpha=0.7)
            ax[0].scatter([pick2[1]], [pick2[0]], color='orange', marker='+', s=s, label='pick 2', alpha=0.7)
            ax[0].scatter([place_v1], [place_u1], color='blue', label='place 1', s=s, alpha=0.7)
            ax[0].scatter([place_v2], [place_u2], color='green', label='place 2', s=s, alpha=0.7)
            ax[1].imshow(rgb_goal, vmin=0.6, vmax=0.7)
            ax[1].imshow(im2, vmin=0.6, vmax=0.7, alpha=0.5)

            if not self.args.noflow:
                flow_im = flow.detach().squeeze().permute(1, 2, 0).cpu().numpy()
                skip = 12
                h, w, _ = flow_im.shape
                ax[2].imshow(np.zeros((h, w)), alpha=0.5)
                ys, xs, _ = np.where(flow_im != 0)
                ax[2].quiver(xs[::skip], ys[::skip],
                            flow_im[ys[::skip], xs[::skip], 1], flow_im[ys[::skip], xs[::skip], 0], 
                            alpha=0.8, color='white', angles='xy', scale_units='xy', scale=1)
            else:
                flow_im = None

            plt.legend()
            plt.tight_layout()
            if 'savefig_path' in dbg_info:
                savefig_path = dbg_info['savefig_path']
            else:
                base_path = '/home/tweng/ws1/src/bimanual_folding/bimanual_ros/scripts/output/{}'.format(test_dir)
                if not os.path.exists(base_path):
                    os.mkdir(base_path)
                savefig_path = '{}/{}.png'.format(base_path, goal_name)
            plt.savefig(savefig_path)
            plt.show()

            data = {
                'rgb_obs': rgb_obs,
                'depth_obs': depth_obs,
                'rgb_goal': rgb_goal,
                'depth_goal': depth_goal,
                'flow': flow_im,
                'action': [pick1, pick2, place1, place2, pred1, pred2],
                'savefig_path': savefig_path
            }
            np.save(savefig_path.replace('png', 'npy'), data)

        return pick1, pick2, place1, place2, data

if __name__ == '__main__':
    rospy.init_node('command_pick_place')

    parser = argparse.ArgumentParser()
    parser.add_argument('--clothtype', default='real_high')
    parser.add_argument('--noflow', action='store_true')
    parser.add_argument('--pickwt1', default="first_265000") # first_300000_noflowpick
    parser.add_argument('--pickwt2', default="second_265000") # second_300000_noflowpick
    parser.add_argument('--placewt1', default="first_300000_noflowplace")
    parser.add_argument('--placewt2', default="second_300000_noflowplace")
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    method = rospy.get_param('method')
    # if method == 'manual':
    m = ManualCommand()
    # elif method == 'FabricFlowNet':
    #     flow_wt = '613'
    #     # pick_wt = '265000' # flow only with spatial aug
    #     m = FabricFlowNet(flow_wt, args)
    # else:
    #     raise NotImplementedError

    pps = PickPlaceServer(m)
    pps.spin()
    