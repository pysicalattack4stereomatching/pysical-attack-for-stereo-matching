import os
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import re

# read all lines in a file
def read_all_lines(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    return lines


# read an .pfm file into numpy array, used to load SceneFlow disparity files
def pfm_imread(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale

class SceneFlowDatset(Dataset):
    def __init__(self, datapath, list_filename, training=False,transform=None,example=0,N=200):
        self.datapath = datapath
        self.left_filenames, self.right_filenames, self.disp_filenames = self.load_path(list_filename)
        self.training = training
        self.transform = transform
        self.start = max(0, min(example, N))
        if example < 0:
            self.N = 1
        else:
            self.N = N-example

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        disp_images = [x[2] for x in splits]
        return left_images, right_images, disp_images

    def load_image(self, filename):
        return Image.open(filename).convert('RGB')

    def load_disp(self, filename):
        data, scale = pfm_imread(filename)
        data = np.ascontiguousarray(data, dtype=np.float32)
        return data

    def __len__(self):
        # return len(self.left_filenames)
        return self.N

    def __getitem__(self, index):
        left_img = self.load_image(os.path.join(self.datapath, self.left_filenames[index]))
        right_img = self.load_image(os.path.join(self.datapath, self.right_filenames[index]))
        disparity = self.load_disp(os.path.join(self.datapath, self.disp_filenames[index]))

        w, h = left_img.size
        crop_w, crop_h = 960, 512

        left_img = left_img.crop((w - crop_w, h - crop_h, w, h))
        right_img = right_img.crop((w - crop_w, h - crop_h, w, h))
        disparity = disparity[h - crop_h:h, w - crop_w: w]

        left_img = np.array(left_img).astype(np.float32)
        right_img = np.array(right_img).astype(np.float32)
        
        if self.transform is not None:
            in_h, in_w, _ = left_img.shape
            imgs = self.transform([left_img]  + [right_img])
            tgt_img = imgs[0]
            
            ref_img_future = imgs[1]
            _, out_h, out_w = tgt_img.shape
        

        return  tgt_img, ref_img_future, disparity, 0, str(index).zfill(6)+'_10.png'
