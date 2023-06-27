# Taken from https://github.com/ClementPinard/SfmLearner-Pytorch/

import torch.utils.data as data
import numpy as np
from PIL import Image
from path import Path
import random
import os

def matchstereo(folders_list02,folders_list03):
    sequence_set = []
    ii=0
    for folder in sorted(folders_list02):
        
        imgs02,imgs03=[],[]
        for img in sorted(os.listdir(folder)):
            if '.jpg' in img:
               imgs02.append(folder+'/'+img)
        for img in sorted(os.listdir(folders_list03[ii])):
            if '.jpg' in img:
               imgs03.append(folders_list03[ii]+'/'+img)
        ii+=1
        for i in range (len(imgs02)):
            sample ={'left':imgs02[i],'right':imgs03[i]}
            sequence_set.append(sample)
    random.shuffle(sequence_set)
    return sequence_set
        
def crawl_folders(folders_list, sequence_length):
        sequence_set = []
        demi_length = (sequence_length-1)//2
        for folder in folders_list:
            intrinsics = np.genfromtxt(folder/'cam.txt', delimiter=',').astype(np.float32).reshape((3, 3))
            imgs = sorted(folder.files('*.jpg'))
            if len(imgs) < sequence_length:
                continue
            for i in range(demi_length, len(imgs)-demi_length):
                sample = {'intrinsics': intrinsics, 'tgt': imgs[i], 'ref_imgs': []}
                for j in range(-demi_length, demi_length + 1):
                    if j != 0:
                        sample['ref_imgs'].append(imgs[i+j])
                sequence_set.append(sample)
        random.shuffle(sequence_set)
        return sequence_set


def load_as_float(path):
    return np.array(Image.open(path)).astype(np.float32)


class SequenceFolder(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000001.jpg
        ..
        root/scene_1/cam.txt
        root/scene_2/0000000.jpg
        .
        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """

    def __init__(self, root, seed=None, train=True, sequence_length=3, transform=None, target_transform=None):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        scene_list_path = self.root/'train.txt' if train else self.root/'val.txt'
        self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
        self.samples = crawl_folders(self.scenes, sequence_length)
        self.transform = transform

    def __getitem__(self, index):
        sample = self.samples[index]
        tgt_img = load_as_float(sample['tgt'])
        ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]
        if self.transform is not None:
            imgs = self.transform([tgt_img] + ref_imgs)
            tgt_img = imgs[0]
            ref_img = imgs[1:]

        return tgt_img, ref_img

    def __len__(self):
        return len(self.samples)

class KittiRawDataLoader(data.Dataset):
    def __init__(self, root, seed=None, train=True, transform=None):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        scene_list_path02 = []
        scene_list_path03 = []
        self.transform = transform
        for filename in sorted(os.listdir(root)):
            A=filename[-2:]
            if filename[-2:]=='02':
                scene_list_path02.append(filename)
            if filename[-2:]=='03':
                scene_list_path03.append(filename) 
        self.scenes02 = [root+'/'+folder for folder in sorted(scene_list_path02)]
        self.scenes03 = [root+'/'+folder for folder in sorted(scene_list_path03)]
        self.samples = matchstereo(self.scenes02,self.scenes03)
        self.N=len(self.samples)
        
    def __getitem__(self, index):
        sample = self.samples[index]
        tgt_img = load_as_float(sample['left'])
        ref_imgs = load_as_float(sample['right'])
        if self.transform is not None:
            imgs = self.transform([tgt_img] + [ref_imgs])
            tgt_img = imgs[0]
            ref_img = imgs[1]
        return tgt_img, ref_img
    def __len__(self):
        return self.N
        