# Adapted from https://github.com/ClementPinard/SfmLearner-Pytorch/

import os
from json import load

import cv2
import numpy as np
import torch
import torch.utils.data as data
from flowutils import flow_io
from path import Path
from PIL import Image
from raw import *
from skimage import transform as sktransform


def load_as_float(path):
    return np.array(Image.open(path)).astype(np.float32)

class ValidationFlowKitti2015MV(data.Dataset):
    """
        Kitti 2015 flow loader
        transform functions must take in a list a images and a numpy array which can be None
    """

    def __init__(self, root, transform=None, N=4000, phase='training', compression=0, raw_root=None, example=0, true_motion=False):
        self.root = Path(root)
        self.start = max(0, min(example, N))
        if example > 0:
            self.N = 1
        else:
            self.N = N

        self.transform = transform
        self.phase = phase
        self.compression = compression
        self.raw_root = raw_root

    def __getitem__(self, index):
        index = self.start + index
        scene = index // 20
        frame = index % 20

        tgt_img_path = self.root.joinpath('data_scene_flow_multiview', self.phase, 'image_2',str(scene).zfill(6)+'_'+str(frame).zfill(2)+'.png')
        ref_img_past_path = self.root.joinpath('data_scene_flow_multiview', self.phase, 'image_2',str(scene).zfill(6)+'_'+str(frame-1).zfill(2)+'.png')
        ref_img_future_path = self.root.joinpath('data_scene_flow_multiview', self.phase, 'image_2',str(scene).zfill(6)+'_'+str(frame+1).zfill(2)+'.png')
        gt_flow_path = self.root.joinpath('data_scene_flow', self.phase, 'flow_occ', str(scene).zfill(6)+'_'+str(frame).zfill(2)+'.png')
        gt_disp_path = self.root.joinpath('data_scene_flow', self.phase, 'disp_occ_0', str(scene).zfill(6)+'_'+str(frame).zfill(2)+'.png')

        tgt_img = load_as_float(tgt_img_path)
        ref_img_future = load_as_float(ref_img_future_path)
        if os.path.exists(gt_flow_path):
            ref_img_past = load_as_float(ref_img_past_path)
        else:
            ref_img_past = torch.zeros(tgt_img.shape)

        he, wi, ch = tgt_img.shape

        gtFlow = None
        if os.path.exists(gt_flow_path):
            u,v,valid = flow_io.flow_read_png(str(gt_flow_path))
            gtFlow = np.dstack((u,v,valid))
            gtFlow = torch.FloatTensor(gtFlow.transpose(2,0,1))
        else:
            gtFlow = torch.zeros((3, he, wi))

        # read disparity
        gtDisp = None
        if os.path.exists(gt_flow_path):
            gtDisp = load_as_float(gt_disp_path)
            gtDisp = np.array(gtDisp,  dtype=float) / 256.
        else:
            gtDisp = torch.zeros((he, wi, 1))

        calib = {}
        poses = {}

        if self.transform is not None:
            in_h, in_w, _ = tgt_img.shape
            imgs = self.transform([tgt_img] + [ref_img_past] + [ref_img_future])
            tgt_img = imgs[0]
            ref_img_past = imgs[1]
            ref_img_future = imgs[2]
            _, out_h, out_w = tgt_img.shape

        return ref_img_past, tgt_img, ref_img_future, gtFlow, gtDisp, calib, poses

    def __len__(self):
        return self.N

class ValidationStereoKitti2015(data.Dataset):
    """
        Kitti 2015 flow loader
        transform functions must take in a list a images and a numpy array which can be None
    """

    def __init__(self, root, transform=None, N=200, phase='training', compression=0, raw_root=None, example=160, true_motion=False):
        self.root = Path(root)
        self.start = max(0, min(example, N))
        if example < 0:
            self.N = 1
        else:
            self.N = N-example

        self.transform = transform
        self.phase = phase
        self.compression = compression
        self.raw_root = raw_root

        self.mapping = [None] * N
        if true_motion:
            mapping_file_path = os.path.join(raw_root,'train_mapping.txt')
            if os.path.exists(mapping_file_path):
                with open(mapping_file_path) as mapping_file:
                    lines = mapping_file.readlines()
                    for i, line in enumerate(lines):
                        if line.strip():
                            split = line.split(' ')
                            self.mapping[i] = {'Scene': '', 'Sequence': '', 'Frame': None}
                            self.mapping[i]['Scene'] = split[0]
                            self.mapping[i]['Sequence'] = split[1]
                            self.mapping[i]['Frame'] = int(split[2].strip())

    def __getitem__(self, index):
        index = self.start + index

        tgt_img_path = self.root.joinpath('data_scene_flow', self.phase, 'image_2',str(index).zfill(6)+'_10.png')
        
        ref_img_future_path = self.root.joinpath('data_scene_flow', self.phase, 'image_3',str(index).zfill(6)+'_10.png')
        
        gt_disp_path = self.root.joinpath('data_scene_flow', self.phase, 'disp_occ_0', str(index).zfill(6)+'_10.png')

        tgt_img = load_as_float(tgt_img_path)
        
        ref_img_future = load_as_float(ref_img_future_path)
        


        # read disparity
        gtDisp = load_as_float(gt_disp_path)
        gtDisp = np.array(gtDisp,  dtype=float) / 256.
        gtDisp = torch.FloatTensor(gtDisp)
        calib = {}
        poses = {}
        # get calibrations
        if self.mapping[index] is not None:
            path = os.path.join(self.raw_root, self.mapping[index]['Scene'])
            seq = self.mapping[index]['Sequence'][len(self.mapping[index]['Scene'] + '_drive')+1:-5]
            dataset = raw(self.raw_root, self.mapping[index]['Scene'], seq, frames=range(self.mapping[index]['Frame'] - 1, self.mapping[index]['Frame'] + 2), origin=1)
            calib = {}
            calib['cam'] = {}
            calib['vel2cam'] = {}
            calib['imu2vel'] = {}
            # import pdb; pdb.set_trace()
            calib['cam']['P_rect_00'] = dataset.calib.P_rect_00
            # calib['cam']['P_rect_00'] = np.eye(4)
            # calib['cam']['P_rect_00'][0, 3] = dataset.calib.P_rect_00[0, 3] / dataset.calib.P_rect_00[0, 0]
            calib['cam']['R_rect_00'] = dataset.calib.R_rect_00
            calib['vel2cam']['RT'] = dataset.calib.T_cam0_velo_unrect
            calib['imu2vel']['RT'] = dataset.calib.T_velo_imu
            poses = [np.array([])] * 3
            poses[0] = dataset.oxts[0].T_w_imu
            poses[1] = dataset.oxts[1].T_w_imu
            poses[2] = dataset.oxts[2].T_w_imu

            
            calib['cam']['baseline'] = dataset.calib.b_rgb

          

        if self.transform is not None:
            in_h, in_w, _ = tgt_img.shape
            imgs = self.transform([tgt_img]  + [ref_img_future])
            tgt_img = imgs[0]
            
            ref_img_future = imgs[1]
            _, out_h, out_w = tgt_img.shape

            # scale projection matrix
            if len(calib) > 0 and (in_h != out_h or in_w != out_w):
                sx = float(out_h) / float(in_h)
                sy = float(out_w) / float(in_w)
                calib['cam']['P_rect_00'][0,0] *= sx
                calib['cam']['P_rect_00'][1,1] *= sy
                calib['cam']['P_rect_00'][0,2] *= sx
                calib['cam']['P_rect_00'][1,2] *= sy

        # set baseline, focal length and principal points
        if len(calib) > 0:
            calib['cam']['focal_length_x'] = calib['cam']['P_rect_00'][0,0]
            calib['cam']['focal_length_y'] = calib['cam']['P_rect_00'][1,1]
            calib['cam']['cx'] = calib['cam']['P_rect_00'][0,2]
            calib['cam']['cy'] = calib['cam']['P_rect_00'][1,2]

            # FROM IMU to IMG00
            calib['P_imu_cam'] = calib['cam']["R_rect_00"].dot(calib['vel2cam']["RT"].dot(calib['imu2vel']["RT"]))
            calib['P_imu_img'] = calib['cam']["P_rect_00"].dot(calib['P_imu_cam'])

        return  tgt_img, ref_img_future, gtDisp, calib, str(index).zfill(6)+'.png'

    def __len__(self):
        return self.N

class ValidationStereoKitti2012(data.Dataset):
    """
        Kitti 2012 flow loader
        transform functions must take in a list a images and a numpy array which can be None
    """

    def __init__(self, root, transform=None, N=194, flow_w=1024, flow_h=384, phase='training', compression=None):
        self.root = Path(root)
        self.N = N
        self.transform = transform
        self.phase = phase
        self.compression = compression
        self.flow_h = flow_h
        self.flow_w = flow_w

    def __getitem__(self, index):
        tgt_img_path =  self.root.joinpath('data_stereo_flow', self.phase, 'colored_0',str(index).zfill(6)+'_10.png')
        ref_img_future_path =  self.root.joinpath('data_stereo_flow', self.phase, 'colored_1',str(index).zfill(6)+'_10.png')
        gt_disp_path = self.root.joinpath('data_stereo_flow', self.phase, 'disp_occ', str(index).zfill(6)+'_10.png')

        tgt_img = load_as_float(tgt_img_path)
        ref_img_future = load_as_float(ref_img_future_path)
        
        # read disparity
        gtDisp = load_as_float(gt_disp_path)
        gtDisp = np.array(gtDisp,  dtype=float) / 256.
        gtDisp = torch.FloatTensor(gtDisp)



        if self.transform is not None:
            in_h, in_w, _ = tgt_img.shape
            imgs = self.transform([tgt_img]  + [ref_img_future])
            tgt_img = imgs[0]
            
            ref_img_future = imgs[1]
            _, out_h, out_w = tgt_img.shape



        return  tgt_img, ref_img_future, gtDisp, 0, str(index).zfill(6)+'_10.png'

    def __len__(self):
        return self.N
    
    
class ValidationStereoKitti2015ADV(data.Dataset):
    """
        Kitti 2015 flow loader
        transform functions must take in a list a images and a numpy array which can be None
    """

    def __init__(self, root, transform=None, N=200, phase='training', compression=0, raw_root=None, example=160, true_motion=False):
        self.root = Path(root)
        self.start = max(0, min(example, N))
        if example < 0:
            self.N = 1
        else:
            self.N = N-example

        self.transform = transform
        self.phase = phase
        self.compression = compression
        self.raw_root = raw_root

        self.mapping = [None] * N
        if true_motion:
            mapping_file_path = os.path.join(raw_root,'train_mapping.txt')
            if os.path.exists(mapping_file_path):
                with open(mapping_file_path) as mapping_file:
                    lines = mapping_file.readlines()
                    for i, line in enumerate(lines):
                        if line.strip():
                            split = line.split(' ')
                            self.mapping[i] = {'Scene': '', 'Sequence': '', 'Frame': None}
                            self.mapping[i]['Scene'] = split[0]
                            self.mapping[i]['Sequence'] = split[1]
                            self.mapping[i]['Frame'] = int(split[2].strip())

    def __getitem__(self, index):
        index = self.start + index

        tgt_img_path = self.root.joinpath('data_scene_flow', self.phase, 'image_2',str(index).zfill(6)+'_10.png')
        
        
        ref_img_future_path = self.root.joinpath('data_scene_flow', self.phase, 'image_3',str(index).zfill(6)+'_10.png')
        
        gt_disp_path = self.root.joinpath('data_scene_flow', self.phase, 'disp_occ_0', str(index).zfill(6)+'_10.png')

        tgt_img = load_as_float(tgt_img_path)
        
        ref_img_future = load_as_float(ref_img_future_path)
        
        ###################
        adv_tgt_img_path = self.root.joinpath("/home/zjc/stereo_attack/stereo_attack/stereo_camouflage_attack/AANet_JPG_20", 'image_2',str(index).zfill(6)+'.jpg')
        adv_ref_img_future_path = self.root.joinpath("/home/zjc/stereo_attack/stereo_attack/stereo_camouflage_attack/AANet_JPG_20", 'image_3',str(index).zfill(6)+'.jpg')
        adv_tgt_img = load_as_float(adv_tgt_img_path)
        adv_ref_img_future =load_as_float(adv_ref_img_future_path)
        
        ###################


        # read disparity
        gtDisp = load_as_float(gt_disp_path)
        gtDisp = np.array(gtDisp,  dtype=float) / 256.
        gtDisp = torch.FloatTensor(gtDisp)
        calib = {}
        poses = {}
        # get calibrations
        if self.mapping[index] is not None:
            path = os.path.join(self.raw_root, self.mapping[index]['Scene'])
            seq = self.mapping[index]['Sequence'][len(self.mapping[index]['Scene'] + '_drive')+1:-5]
            dataset = raw(self.raw_root, self.mapping[index]['Scene'], seq, frames=range(self.mapping[index]['Frame'] - 1, self.mapping[index]['Frame'] + 2), origin=1)
            calib = {}
            calib['cam'] = {}
            calib['vel2cam'] = {}
            calib['imu2vel'] = {}
            # import pdb; pdb.set_trace()
            calib['cam']['P_rect_00'] = dataset.calib.P_rect_00
            # calib['cam']['P_rect_00'] = np.eye(4)
            # calib['cam']['P_rect_00'][0, 3] = dataset.calib.P_rect_00[0, 3] / dataset.calib.P_rect_00[0, 0]
            calib['cam']['R_rect_00'] = dataset.calib.R_rect_00
            calib['vel2cam']['RT'] = dataset.calib.T_cam0_velo_unrect
            calib['imu2vel']['RT'] = dataset.calib.T_velo_imu
            poses = [np.array([])] * 3
            poses[0] = dataset.oxts[0].T_w_imu
            poses[1] = dataset.oxts[1].T_w_imu
            poses[2] = dataset.oxts[2].T_w_imu

            
            calib['cam']['baseline'] = dataset.calib.b_rgb

          

        if self.transform is not None:
            in_h, in_w, _ = tgt_img.shape
            imgs = self.transform([tgt_img]  + [ref_img_future]+[adv_tgt_img]+[adv_ref_img_future])
            tgt_img = imgs[0]
            
            ref_img_future = imgs[1]
            
            adv_tgt_img = imgs[2]
            adv_ref_img_future=imgs[3]
            
            _, out_h, out_w = tgt_img.shape

            # scale projection matrix
            if len(calib) > 0 and (in_h != out_h or in_w != out_w):
                sx = float(out_h) / float(in_h)
                sy = float(out_w) / float(in_w)
                calib['cam']['P_rect_00'][0,0] *= sx
                calib['cam']['P_rect_00'][1,1] *= sy
                calib['cam']['P_rect_00'][0,2] *= sx
                calib['cam']['P_rect_00'][1,2] *= sy

        # set baseline, focal length and principal points
        if len(calib) > 0:
            calib['cam']['focal_length_x'] = calib['cam']['P_rect_00'][0,0]
            calib['cam']['focal_length_y'] = calib['cam']['P_rect_00'][1,1]
            calib['cam']['cx'] = calib['cam']['P_rect_00'][0,2]
            calib['cam']['cy'] = calib['cam']['P_rect_00'][1,2]

            # FROM IMU to IMG00
            calib['P_imu_cam'] = calib['cam']["R_rect_00"].dot(calib['vel2cam']["RT"].dot(calib['imu2vel']["RT"]))
            calib['P_imu_img'] = calib['cam']["P_rect_00"].dot(calib['P_imu_cam'])

        return  tgt_img, ref_img_future, gtDisp, adv_tgt_img,adv_ref_img_future, str(index).zfill(6)+'.png'

    def __len__(self):
        return self.N
    
class ValidationZED(data.Dataset):
    """
        ZED loader
        transform functions must take in a list a images and a numpy array which can be None
    """

    def __init__(self, root, transform=None, N=12, flow_w=1024, flow_h=384, phase='training', compression=None):
        self.root = Path(root)
        self.N = N
        self.transform = transform
        self.phase = phase
        self.compression = compression
        self.flow_h = flow_h
        self.flow_w = flow_w

    def __getitem__(self, index):
        
        tgt_img_path =  self.root.joinpath('ZED24_original', 'image_2',str(index+1).zfill(4)+'_01_left.png')
        ref_img_future_path =  self.root.joinpath('ZED24_original', 'image_3',str(index+1).zfill(4)+'_01_right.png')
        
        patch_img_path =  self.root.joinpath('ZED24_patch', 'image_2',str(index+1).zfill(4)+'_00_left.png')
        patch_img_future_path =  self.root.joinpath('ZED24_patch', 'image_3',str(index+1).zfill(4)+'_00_right.png')
        

        tgt_img = load_as_float(tgt_img_path)
        ref_img_future = load_as_float(ref_img_future_path)
        
        patch_img = load_as_float(patch_img_path)
        patch_img_future = load_as_float(patch_img_future_path)
        
        



        if self.transform is not None:
            in_h, in_w, _ = tgt_img.shape
            imgs = self.transform([tgt_img]  + [ref_img_future]+[patch_img]+[patch_img_future])
            tgt_img = imgs[0]
            ref_img_future = imgs[1]
            patch_img = imgs[2]
            patch_img_future = imgs[3]
            _, out_h, out_w = tgt_img.shape



        return  tgt_img, ref_img_future,patch_img,patch_img_future,str(index+1).zfill(6)+'.png'

    def __len__(self):
        return self.N

class ValidationStereoKittiRawData(data.Dataset):
    """
        Kitti 2015 flow loader
        transform functions must take in a list a images and a numpy array which can be None
    """

    def __init__(self, root, transform=None, N=200, phase='training', compression=0, raw_root=None, example=160, true_motion=False):
        self.root = Path(root)
        self.start = max(0, min(example, N))
        if example < 0:
            self.N = 1
        else:
            self.N = N-example

        self.transform = transform
        self.phase = phase
        self.compression = compression
        self.raw_root = raw_root

        self.mapping = [None] * N
        if true_motion:
            mapping_file_path = os.path.join(raw_root,'train_mapping.txt')
            if os.path.exists(mapping_file_path):
                with open(mapping_file_path) as mapping_file:
                    lines = mapping_file.readlines()
                    for i, line in enumerate(lines):
                        if line.strip():
                            split = line.split(' ')
                            self.mapping[i] = {'Scene': '', 'Sequence': '', 'Frame': None}
                            self.mapping[i]['Scene'] = split[0]
                            self.mapping[i]['Sequence'] = split[1]
                            self.mapping[i]['Frame'] = int(split[2].strip())

    def __getitem__(self, index):
        index = self.start + index

        # tgt_img_path = self.root.joinpath('data_scene_flow', self.phase, 'image_2',str(index).zfill(6)+'_10.png')
        
        # ref_img_future_path = self.root.joinpath('data_scene_flow', self.phase, 'image_3',str(index).zfill(6)+'_10.png')
        
        # gt_disp_path = self.root.joinpath('data_scene_flow', self.phase, 'disp_occ_0', str(index).zfill(6)+'_10.png')
        
        tgt_img_path = self.root.joinpath("/home/zjc/kitti/2011_09_30/2011_09_30_drive_0016_sync",  'image_02','data',str(index).zfill(10)+'.png')
        
        ref_img_future_path = self.root.joinpath("/home/zjc/kitti/2011_09_30/2011_09_30_drive_0016_sync", 'image_03','data',str(index).zfill(10)+'.png')

        tgt_img = load_as_float(tgt_img_path)
        
        ref_img_future = load_as_float(ref_img_future_path)
        


        # read disparity
        # gtDisp = load_as_float(gt_disp_path)
        # gtDisp = np.array(gtDisp,  dtype=float) / 256.
        # gtDisp = torch.FloatTensor(gtDisp)
        gtDisp =0
        calib = {}
        poses = {}
        # get calibrations

          

        if self.transform is not None:
            in_h, in_w, _ = tgt_img.shape
            imgs = self.transform([tgt_img]  + [ref_img_future])
            tgt_img = imgs[0]
            
            ref_img_future = imgs[1]
            _, out_h, out_w = tgt_img.shape

            # scale projection matrix
            

        # set baseline, focal length and principal points
        

        return  tgt_img, ref_img_future, gtDisp, calib, str(index).zfill(10)+'.png'

    def __len__(self):
        return self.N
    
class ValidationStereoKitti2015WithSemantic(data.Dataset):
    """
        Kitti 2015 flow loader
        transform functions must take in a list a images and a numpy array which can be None
    """

    def __init__(self, root, transform=None, N=200, phase='training', compression=0, raw_root=None, example=160, true_motion=False):
        self.root = Path(root)
        self.start = max(0, min(example, N))
        if example < 0:
            self.N = 1
        else:
            self.N = N-example

        self.transform = transform
        self.phase = phase
        self.compression = compression
        self.raw_root = raw_root

        self.mapping = [None] * N
        if true_motion:
            mapping_file_path = os.path.join(raw_root,'train_mapping.txt')
            if os.path.exists(mapping_file_path):
                with open(mapping_file_path) as mapping_file:
                    lines = mapping_file.readlines()
                    for i, line in enumerate(lines):
                        if line.strip():
                            split = line.split(' ')
                            self.mapping[i] = {'Scene': '', 'Sequence': '', 'Frame': None}
                            self.mapping[i]['Scene'] = split[0]
                            self.mapping[i]['Sequence'] = split[1]
                            self.mapping[i]['Frame'] = int(split[2].strip())

    def __getitem__(self, index):
        index = self.start + index
        
        tgt_img_path = self.root.joinpath('data_scene_flow', self.phase, 'image_2',str(index).zfill(6)+'_10.png')
        
        ref_img_future_path = self.root.joinpath('data_scene_flow', self.phase, 'image_3',str(index).zfill(6)+'_10.png')
        
        gt_disp_path = self.root.joinpath('data_scene_flow', self.phase, 'disp_occ_0', str(index).zfill(6)+'_10.png')

        tgt_img = load_as_float(tgt_img_path)
        
        ref_img_future = load_as_float(ref_img_future_path)
        
        tgt_img_semantic_path = self.root.joinpath('data_scene_flow', self.phase, 'semantic',str(index).zfill(6)+'_10.png')
        
        tgt_img_semantic = load_as_float(tgt_img_semantic_path)
        
        # classsemantic ={}
        # #求掩膜
        # imgmaskinit = tgt_img_semantic
        # xy = np.where(imgmaskinit==26) #汽车掩膜
        # imgout_car = np.zeros_like(imgmaskinit)
        # imgout_car[xy]=1
        
        # xy = np.where(imgmaskinit==23) #天空掩膜
        # imgout_sky = np.zeros_like(imgmaskinit)
        # imgout_sky[xy]=1
        
        # xy = np.where(imgmaskinit==7) #道路掩膜
        # imgout_road = np.zeros_like(imgmaskinit)
        # imgout_road[xy]=1
        
        # xy = np.where(imgmaskinit==24) #行人掩膜
        # imgout_person = np.zeros_like(imgmaskinit)
        # imgout_person[xy]=1
        
        # xy = np.where(imgmaskinit==21) #植被掩膜
        # imgout_vegetation = np.zeros_like(imgmaskinit)
        # imgout_vegetation[xy]=1
        
        
        
        # read disparity
        gtDisp = load_as_float(gt_disp_path)
        gtDisp = np.array(gtDisp,  dtype=float) / 256.
        gtDisp = torch.FloatTensor(gtDisp)
        calib = {}
        poses = {}
        # get calibrations
        if self.mapping[index] is not None:
            path = os.path.join(self.raw_root, self.mapping[index]['Scene'])
            seq = self.mapping[index]['Sequence'][len(self.mapping[index]['Scene'] + '_drive')+1:-5]
            dataset = raw(self.raw_root, self.mapping[index]['Scene'], seq, frames=range(self.mapping[index]['Frame'] - 1, self.mapping[index]['Frame'] + 2), origin=1)
            calib = {}
            calib['cam'] = {}
            calib['vel2cam'] = {}
            calib['imu2vel'] = {}
            # import pdb; pdb.set_trace()
            calib['cam']['P_rect_00'] = dataset.calib.P_rect_00
            # calib['cam']['P_rect_00'] = np.eye(4)
            # calib['cam']['P_rect_00'][0, 3] = dataset.calib.P_rect_00[0, 3] / dataset.calib.P_rect_00[0, 0]
            calib['cam']['R_rect_00'] = dataset.calib.R_rect_00
            calib['vel2cam']['RT'] = dataset.calib.T_cam0_velo_unrect
            calib['imu2vel']['RT'] = dataset.calib.T_velo_imu
            poses = [np.array([])] * 3
            poses[0] = dataset.oxts[0].T_w_imu
            poses[1] = dataset.oxts[1].T_w_imu
            poses[2] = dataset.oxts[2].T_w_imu

            
            calib['cam']['baseline'] = dataset.calib.b_rgb

          

        if self.transform is not None:
            in_h, in_w, _ = tgt_img.shape
            imgs = self.transform([tgt_img]  + [ref_img_future])
            tgt_img = imgs[0]
            
            ref_img_future = imgs[1]
            # ##掩膜变换
            # imgsemantic = imgs[2:]
            # imgsemantic[imgsemantic>0.5]=1
            # imgsemantic[imgsemantic<=0.5]=0
            # imgout_car =imgsemantic[0]
            # imgout_sky = imgsemantic[1]
            # imgout_road =imgsemantic[2]
            # imgout_person=imgsemantic[3]
            # imgout_vegetation = imgsemantic[4]
            
            # classsemantic['class_car_area'] = np.mean(imgout_car)*100 # 求掩膜区域占比
            # classsemantic['class_sky_area'] = np.mean(imgout_sky)*100 # 求掩膜区域占比
            # classsemantic['class_road_area'] = np.mean(imgout_road)*100 # 求掩膜区域占比
            # classsemantic['class_person_area'] = np.mean(imgout_person)*100 # 求掩膜区域占比
            # classsemantic['class_vege_area'] = np.mean(imgout_vegetation)*100 # 求掩膜区域占比
            
                      
            _, out_h, out_w = tgt_img.shape

            # scale projection matrix
            if len(calib) > 0 and (in_h != out_h or in_w != out_w):
                sx = float(out_h) / float(in_h)
                sy = float(out_w) / float(in_w)
                calib['cam']['P_rect_00'][0,0] *= sx
                calib['cam']['P_rect_00'][1,1] *= sy
                calib['cam']['P_rect_00'][0,2] *= sx
                calib['cam']['P_rect_00'][1,2] *= sy

        # set baseline, focal length and principal points
        if len(calib) > 0:
            calib['cam']['focal_length_x'] = calib['cam']['P_rect_00'][0,0]
            calib['cam']['focal_length_y'] = calib['cam']['P_rect_00'][1,1]
            calib['cam']['cx'] = calib['cam']['P_rect_00'][0,2]
            calib['cam']['cy'] = calib['cam']['P_rect_00'][1,2]

            # FROM IMU to IMG00
            calib['P_imu_cam'] = calib['cam']["R_rect_00"].dot(calib['vel2cam']["RT"].dot(calib['imu2vel']["RT"]))
            calib['P_imu_img'] = calib['cam']["P_rect_00"].dot(calib['P_imu_cam'])

        #return  tgt_img, ref_img_future, gtDisp, calib, poses,imgout_car,imgout_sky,imgout_road,imgout_person,imgout_vegetation,classsemantic
        return  tgt_img, ref_img_future, gtDisp, calib, poses,tgt_img_semantic
    def __len__(self):
        return self.N