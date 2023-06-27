
import argparse
import time
import csv
import datetime
import cv2

import numpy as np
from scipy.ndimage import rotate, zoom

import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn as nn
import torch.utils.data
from datasets.sequence_folders import SequenceFolder,KittiRawDataLoader
from datasets.sceneflow_dataset import SceneFlowDatset
import custom_transforms
import models
from models.stereo_models import StereoModel
from utils import *
from logger import TermLogger, AverageMeter
from path import Path
from itertools import chain
from tensorboardX import SummaryWriter
from losses import compute_epe, compute_cossim, multiscale_cossim,D1_metric,get_loss

import torch.nn.functional as F

import scipy.signal as signal

epsilon = 1e-8

parser = argparse.ArgumentParser(description='Generating Adversarial Patches for Stereo Matching Networks',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data', dest='data', default="/home/zjc/stereo_attack/stereo_attack/flowattack-master/path/to/resulting",
                    help='path to dataset')
parser.add_argument('--kitti-data', dest='kitti_data', default="/home/zjc/kitti/",
                    help='path to kitti dataset')
parser.add_argument('--sceneflow-data', dest='sceneflow_data', default="/home/zjc/dataset/",
                    help='path to sceneflow dataset')
parser.add_argument('--patch-path', dest='patch_path', default='/home/zjc/stereo_attack/stereo_attack/stereo_camouflage_attack/checkpoints/demo/AANet/pgd-checker-init-cCV-patch0_3/patch_epoch_5',
                    help='Initialize patch from here')
parser.add_argument('--mask-path', dest='mask_path', default='',
                    help='Initialize mask from here')
parser.add_argument('--valset', dest='valset', type=str, default='kitti2015', choices=['kitti2015', 'kitti2012','sceneflow'],
                    help='Optical stereo validation dataset')
parser.add_argument('--DEBUG', action='store_true', help='DEBUG Mode')
parser.add_argument('--name', dest='name', type=str, default='demo', required=False,
                    help='name of the experiment, checpoints are stored in checpoints/name')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate-1e3,if gd,use rate 10 if pgdattack,use rate 1e-3')
parser.add_argument('--epochs', default=1, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--max-count', default=5, type=int,
                    help='max count')
parser.add_argument('--epoch-size', default=1000, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if not set)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                    help='beta parameters for adam')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--stereonet', dest='stereonet', type=str, default='PSMNet', choices=['PSMNet','AANet', 'sttr'],
                    help='stereo network architecture.')
parser.add_argument('--alpha', default=0.0, type=float, help='regularization weight')
parser.add_argument('--image-size', type=int, default=384, help='the min(height, width) of the input image to network')
parser.add_argument('--patch-type', type=str, default='square', help='patch type: circle or square,checker or random')
parser.add_argument('--patch-size', type=float, default=0.20, help='patch size. E.g. 0.05 ~= 5% of image ')

parser.add_argument('--attack_mode', default='cost_volume_ave', type=str, help="attack mode [cost_volume_ave or zero]")

parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--log-summary', default='progress_log_summary.csv', metavar='PATH',
                    help='csv where to save per-epoch train and valid stats')
parser.add_argument('--log-full', default='progress_log_full.csv', metavar='PATH',
                    help='csv where to save per-gradient descent train stats')
parser.add_argument('--log-output', type=bool, default=True, help='will log dispnet outputs and warped imgs at validation step')
parser.add_argument('--norotate', action='store_true', help='will not apply rotation augmentation')
parser.add_argument('--log-terminal', action='store_true', help='will display progressbar at terminal')
parser.add_argument('-f', '--training-output-freq', type=int, help='frequence for outputting dispnet outputs and warped imgs at training for all scales if 0 will not output',
                    metavar='N', default=50)

best_error = -1
n_iter = 0


def main():
    global args, best_error, n_iter
    args = parser.parse_args()
    save_path = Path(args.name)
    args.save_path = 'checkpoints'/save_path #/timestamp
    print('=> will save everything to {}'.format(args.save_path))
    args.save_path.makedirs_p()
    torch.manual_seed(args.seed)

    training_writer = SummaryWriter(args.save_path)
    output_writer = SummaryWriter(args.save_path/'valid')

    # Data loading code
    stereo_loader_h, stereo_loader_w = 384, 1280

    train_transform = custom_transforms.Compose([
        custom_transforms.RandomHorizontalFlip(),
        custom_transforms.RandomScaleCrop(h=stereo_loader_h, w=512),
        custom_transforms.ArrayToTensor(),
        ])

    valid_transform = custom_transforms.Compose([custom_transforms.Scale(h=stereo_loader_h, w=stereo_loader_w),
                            custom_transforms.ArrayToTensor()])

    print("=> fetching scenes in '{}'".format(args.data))
    train_set = KittiRawDataLoader(args.data,
        transform=train_transform,
        seed=args.seed,
        train=True,
        )
    

    if args.valset =="kitti2015":
        from datasets.validation_stereo import ValidationStereoKitti2015
        val_set = ValidationStereoKitti2015(root=args.kitti_data, transform=valid_transform,example=0)
    elif args.valset =="kitti2012":
        from datasets.validation_stereo import ValidationStereoKitti2012
        val_set = ValidationStereoKitti2012(root=args.kitti_data, transform=valid_transform)
    elif args.valset == "sceneflow":
        valid_transform = custom_transforms.Compose([custom_transforms.Scale(h=512, w=960),
                            custom_transforms.ArrayToTensor()])
        val_set = SceneFlowDatset(datapath=args.sceneflow_data,list_filename='datasets/scene_test.txt',transform=valid_transform)

    if args.DEBUG:
        train_set.__len__ = 32
        train_set.samples = train_set.samples[:32]

    print('{} samples found in {} train scenes'.format(len(train_set), len(train_set.samples)))
    print('{} samples found in valid scenes'.format(len(val_set)))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.workers, pin_memory=True, drop_last=True)

    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1,               # batch size is 1 since images in kitti have different sizes
                    shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=True)

    if args.epoch_size == 0:
        args.epoch_size = len(train_loader)

    # create model
    print("=> creating model")

    
    if args.stereonet =='PSMNet':
        stereo_net = StereoModel(method='psmnet')
        print("=> using pre-trained weights for PSMNet")
        stereo_net.restore_model('pretrained_stereo_models/psmnet/pretrained_KITTI2015_finetuned.pth')
        
    elif args.stereonet == 'AANet':
        stereo_net = StereoModel(method='aanet')
        print("=> using pre-trained weights for AANet")
        stereo_net.restore_model('pretrained_stereo_models/aanet/aanet_kitti15-finetuned.pth')
        
    elif args.stereonet == 'sttr':
        stereo_net = StereoModel(method='sttr')
        print("=> using pre-trained weights for StereoTransformer")
        stereo_net.restore_model('pretrained_stereo_models/sttr/kitti_finetuned_model.pth.tar')
        
    else:
        stereo_net = getattr(models, args.stereonet)()
        stereo_net.init_weights()

    pytorch_total_params = sum(p.numel() for p in stereo_net.parameters())
    print("Number of model paramters: " + str(pytorch_total_params))

    cudnn.benchmark = True
    
    if args.patch_type == 'checker':
        checker_img =checker_produce(int(args.image_size*args.patch_size),int(args.image_size*args.patch_size),16)
        patch =np.expand_dims ((np.array(checker_img)/255.0).transpose(2,0,1),axis=0)
        patch_shape = patch.shape
        patch_init = patch.copy()
        mask = np.ones(patch_shape)
    elif args.patch_type == 'circle':
        patch, mask, patch_shape = init_patch_circle(args.image_size, args.patch_size,args.batch_size)
        patch_init = patch.copy()
    elif args.patch_type == 'square':
        # patch, patch_shape = init_patch_square(args.image_size, args.patch_size,args.batch_size)
        patch, patch_shape = init_patch_square(args.image_size, args.patch_size,args.batch_size,1)
        patch_init = patch.copy()
        mask = np.ones(patch_shape)
    else:
        sys.exit("Please choose a square or circle patch")
    preepoch=0
    if args.patch_path:
        print("Loading patch from ", args.patch_path)
        patch_epoch_restore = torch.load(args.patch_path)
        patch=patch_epoch_restore['patch']
        #preepoch =  patch_epoch_restore['epoch']
        print('We start from {} Epoch'.format(preepoch))   
        patch_init = patch.copy()
        patch_shape = patch.shape
        if args.mask_path:
            mask_image = load_as_float(args.mask_path)
            mask_image = np.array(Image.fromarray(mask_image).resize((patch_shape[-1], patch_shape[-2])))/256.
            #mask_image = imresize(mask_image, (patch_shape[-1], patch_shape[-2]))/256.
            mask = np.array([mask_image.transpose(2,0,1)])
        else:
            if args.patch_type == 'circle':
                mask = createCircularMask(patch_shape[-2], patch_shape[-1]).astype('float32')
                mask = np.array([[mask,mask,mask]])
            elif args.patch_type == 'square':
                mask = np.ones(patch_shape)
 

    if args.log_terminal:
        logger = TermLogger(n_epochs=args.epochs, train_size=min(len(train_loader), args.epoch_size), valid_size=len(val_loader), attack_size=args.max_count)
        logger.epoch_bar.start()
    else:
        logger=None
    error_names = ['epe', 'adv_epe', 'PeoD1_all', 'adv_cos_sim','D1_all','adv_D1_all']
    errors =[0 for i in range(len(error_names))]
    
    dirs = args.stereonet+"Prediction"
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    
    
    for epoch in range(preepoch,args.epochs):

        if args.log_terminal:
            logger.epoch_bar.update(epoch)
            logger.reset_train_bar()
        
        # Validate
                
        
        if epoch %1 == 0:
            errors, error_names = validate_stereo_with_gt(patch, mask, patch_shape, val_loader, stereo_net,epoch, logger, output_writer,errors,error_names,dirs)

            error_string = ', '.join('{} : {:.3f}'.format(name, error) for name, error in zip(error_names, errors))
        #
        for error, name in zip(errors, error_names):
            training_writer.add_scalar(name, error, epoch)
            
        # error_string='0'
        # print('Epoch {} completed,loss_train{}'.format(epoch,loss_ave))
        
        if args.log_terminal:
            logger.valid_writer.write(' * Avg {}'.format(error_string))
        else:
            loss_ave =0
            print('Epoch {} completed,loss_train{}'.format(epoch,loss_ave))

        
        
        patch_epoch={'patch':patch,'epoch':epoch}
        torch.save(patch_epoch, args.save_path/'patch_epoch_{}'.format(str(epoch)))

    if args.log_terminal:
        logger.epoch_bar.finish()


def train(patch, mask, patch_init, patch_shape, train_loader, stereo_net ,epoch, logger=None, train_writer=None,target_stereo = 'cost_volume_ave'):
    global args, n_iter
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter(precision=4)
    stereo_net.eval()

    end = time.time()
    loss_ave=[]

    patch_shape_orig = patch_shape
    for i, (tgt_img, ref_img) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)
        tgt_img_var = Variable(tgt_img.cuda())
        ref_right_img_var = Variable(ref_img.cuda())
        
        
        data_shape = tgt_img.cpu().numpy().shape
        

        if args.patch_type == 'circle':
            patch, mask, patch_init, rx, ry, patch_shape = circle_transform(patch, mask, patch_init, data_shape, patch_shape, True,fixed_loc=(200,200))
            
        elif args.patch_type == 'square' or 'checker':
            ref_patch, ref_mask, ref_patch_init, ref_rx, ref_ry = square_transform(patch, mask, patch_init, data_shape, patch_shape,patch_shape, norotate=args.norotate)
            tar_patch, tar_mask, tar_patch_init, rx, ry = square_transform(patch, mask, patch_init, data_shape, patch_shape,patch_shape, norotate=args.norotate,fixed_loc=(ref_rx,ref_ry))
                    
        
        patch, mask = torch.FloatTensor(tar_patch), torch.FloatTensor(tar_mask)
        patch_init = torch.FloatTensor(tar_patch_init)
        
        ref_patch, ref_mask = torch.FloatTensor(ref_patch), torch.FloatTensor(ref_mask)
        ref_patch_init = torch.FloatTensor(ref_patch_init)
        
        
        stereo_pred_var = stereo_net.forward(tgt_img_var, ref_right_img_var,'CV')
        
        

        patch, mask = patch.cuda(), mask.cuda()
        patch_init = patch_init.cuda()
        patch_var, mask_var = Variable(patch), Variable(mask)
        patch_init_var = Variable(patch_init).cuda()
        ###################################
        ref_patch, ref_mask = ref_patch.cuda(), ref_mask.cuda()
        ref_patch_init = ref_patch_init.cuda()
        ref_patch_var, ref_mask_var = Variable(ref_patch), Variable(ref_mask)
        ref_patch_init_var = Variable(ref_patch_init).cuda()
        
        if target_stereo == 'zero':
            target_var = torch.ones_like(stereo_pred_var)
            target_var = Variable (target_var, requires_grad=True).cuda()
            
        elif target_stereo == 'neg_flow':
            target_var = Variable(-1*stereo_pred_var.data.clone(), requires_grad=True).cuda()
        elif target_stereo =='cost_volume_ave':
            
            target_var = torch.zeros_like(stereo_pred_var)
            target_var[:,:,1,:,:]=1         
            target_var=Variable (target_var, requires_grad=True).cuda()
           
        adv_tgt_img_var, adv_ref_right_img_var, patch_var,loss_item = pgdattack(stereo_net, tgt_img_var, ref_right_img_var, patch_var, mask_var, patch_init_var, target_var, logger,ref_patch_var, ref_mask_var,target_stereo=target_stereo,disparity=int(-ref_rx+rx))
                   
        
        masked_patch_var = torch.mul(mask_var, patch_var)
        patch = masked_patch_var.data.cpu().numpy() 
        mask = mask_var.data.cpu().numpy()
        patch_init = patch_init_var.data.cpu().numpy()
        new_patch = np.zeros(patch_shape)
        new_mask = np.zeros(patch_shape)
        new_patch_init = np.zeros(patch_shape)
        for x in range(new_patch.shape[0]):
            for y in range(new_patch.shape[1]):
                new_patch[x][y] = patch[x][y][ry:ry+patch_shape[-2], rx:rx+patch_shape[-1]]
                new_mask[x][y] = mask[x][y][ry:ry+patch_shape[-2], rx:rx+patch_shape[-1]]
                new_patch_init[x][y] = patch_init[x][y][ry:ry+patch_shape[-2], rx:rx+patch_shape[-1]]

        patch = new_patch 
        mask = new_mask
        patch_init = new_patch_init
        loss_ave.append(loss_item)
        
        

        patch = zoom(patch, zoom=(1,1,patch_shape_orig[2]/patch_shape[2], patch_shape_orig[3]/patch_shape[3]), order=1)
        mask = zoom(mask, zoom=(1,1,patch_shape_orig[2]/patch_shape[2], patch_shape_orig[3]/patch_shape[3]), order=0)
        patch_init = zoom(patch_init, zoom=(1,1,patch_shape_orig[2]/patch_shape[2], patch_shape_orig[3]/patch_shape[3]), order=1)
        
        if args.training_output_freq > 0 and n_iter % args.training_output_freq == 0:
            train_writer.add_image('train tgt image', transpose_image(tensor2array(tgt_img[0])), n_iter)
            train_writer.add_image('train ref future image', transpose_image(tensor2array(ref_img[0])), n_iter)
            train_writer.add_image('train adv tgt image', transpose_image(tensor2array(adv_tgt_img_var.data.cpu()[0])), n_iter)
            train_writer.add_image('train adv ref future image', transpose_image(tensor2array(adv_ref_right_img_var.data.cpu()[0])), n_iter)
            train_writer.add_image('train patch', transpose_image(tensor2array(patch_var.data.cpu()[0])), n_iter)
            train_writer.add_image('train patch init', transpose_image(tensor2array(patch_init_var.data.cpu()[0])), n_iter)
            train_writer.add_image('train mask', transpose_image(tensor2array(mask_var.data.cpu()[0])), n_iter)

        
        
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.log_terminal:
            logger.train_bar.update(i+1)
        if i >= args.epoch_size - 1:
            break

        n_iter += 1
    
    return patch, mask, patch_init, patch_shape,np.mean(loss_ave)



def attack(stereo_net, tgt_img_var, ref_right_img_var, patch_var, mask_var, patch_init_var, target_var, logger,ref_patch_var, ref_mask_var,target_stereo='neg_flow',disparity=None):
    global args
    stereo_net.eval()

    adv_tgt_img_var = torch.mul((1-mask_var), tgt_img_var) + torch.mul(mask_var, patch_var)
    
    
    
    adv_ref_right_img_var = torch.mul((1-ref_mask_var), ref_right_img_var) + torch.mul(ref_mask_var, ref_patch_var)
    
    
    count = 0
    loss_scalar = 1
    while loss_scalar > 0.1 :
        count += 1
        adv_tgt_img_var = Variable(adv_tgt_img_var.data, requires_grad = True)
        adv_ref_right_img_var = Variable(adv_ref_right_img_var.data, requires_grad = True)
        just_the_patch = Variable(patch_var.data, requires_grad=True)

        adv_stereo_out_var = stereo_net.forward(adv_tgt_img_var, adv_ref_right_img_var)
        
        ############
        if target_stereo=='neg_flow':
            loss_data = (1 - nn.functional.cosine_similarity(adv_stereo_out_var, target_var )).mean()
            loss_reg = nn.functional.l1_loss(torch.mul(mask_var,just_the_patch), torch.mul(mask_var, patch_init_var))
            
        elif target_stereo=='zero':
            
            loss_data = get_loss(adv_stereo_out_var, target_var,mode='mse')
                      
            loss_reg = nn.functional.l1_loss(torch.mul(mask_var,just_the_patch), torch.mul(mask_var, patch_init_var))
        elif target_stereo=='cost_volume_ave':
            loss_data = get_loss(adv_stereo_out_var, target_var,mode='mse')*10000
            
            
            
            loss_reg = nn.functional.l1_loss(torch.mul(mask_var,just_the_patch), torch.mul(mask_var, patch_init_var))
        
        loss = (1-args.alpha)*loss_data + args.alpha*loss_reg

        loss.backward()

        adv_tgt_img_grad = adv_tgt_img_var.grad.clone()
        
        adv_ref_future_img_grad = adv_ref_right_img_var.grad.clone()

        adv_tgt_img_var.grad.data.zero_()
        adv_ref_right_img_var.grad.data.zero_()
        bs,c,h,w= adv_tgt_img_grad.shape
        
        patch_var += torch.clamp((torch.mean(args.lr*(adv_tgt_img_grad),dim=0)).expand(bs,c,h,w), -0.02, 0.02) 

        adv_tgt_img_var = torch.mul((1-mask_var), tgt_img_var) + torch.mul(mask_var, patch_var) 
        advpad = nn.ZeroPad2d(padding=(0,disparity,0,0))
        ref_ma = advpad(patch_var[:,:,:,disparity:])
        adv_ref_right_img_var = torch.mul((1-ref_mask_var), ref_right_img_var) + torch.mul(ref_mask_var, ref_ma)

        adv_tgt_img_var = torch.clamp(adv_tgt_img_var, -1, 1)

        adv_ref_right_img_var = torch.clamp(adv_ref_right_img_var, -1, 1)

        loss_scalar = loss.item()

        if args.log_terminal:
            logger.attack_bar.update(count)

        if count > args.max_count-1:
            break

    return adv_tgt_img_var, adv_ref_right_img_var, patch_var,loss_scalar

def pgdattack(stereo_net, tgt_img_var, ref_right_img_var, patch_var, mask_var, patch_init_var, target_var, logger,ref_patch_var, ref_mask_var,target_stereo='neg_flow',disparity=None):
    global args
    stereo_net.eval()

    adv_tgt_img_var = torch.mul((1-mask_var), tgt_img_var) + torch.mul(mask_var, patch_var)
    
    
    
    adv_ref_right_img_var = torch.mul((1-ref_mask_var), ref_right_img_var) + torch.mul(ref_mask_var, ref_patch_var)
    
    
    count = 0
    loss_scalar = 1
    while loss_scalar > 0.1 :
        count += 1
        adv_tgt_img_var = Variable(adv_tgt_img_var.data, requires_grad = True)
        adv_ref_right_img_var = Variable(adv_ref_right_img_var.data, requires_grad = True)
        just_the_patch = Variable(patch_var.data, requires_grad=True)

        adv_stereo_out_var = stereo_net.forward(adv_tgt_img_var, adv_ref_right_img_var,'CV')
        
        if target_stereo=='neg_flow':
            loss_data = (1 - nn.functional.cosine_similarity(adv_stereo_out_var, target_var)).mean()
            loss_reg = nn.functional.l1_loss(torch.mul(mask_var,just_the_patch), torch.mul(mask_var, patch_init_var))
           
        elif target_stereo=='zero':
            
            loss_data = get_loss(adv_stereo_out_var, target_var,mode='mse')
                      
            loss_reg = nn.functional.l1_loss(torch.mul(mask_var,just_the_patch), torch.mul(mask_var, patch_init_var))
        elif target_stereo=='cost_volume_ave':
            loss_data = get_loss(adv_stereo_out_var, target_var,mode='mse')*10000
            
            
            loss_reg = nn.functional.l1_loss(torch.mul(mask_var,just_the_patch), torch.mul(mask_var, patch_init_var))
       
        loss = (1-args.alpha)*loss_data + args.alpha*loss_reg

        loss.backward()

        adv_tgt_img_grad = adv_tgt_img_var.grad.clone()
        
        adv_ref_future_img_grad = adv_ref_right_img_var.grad.clone()

        adv_tgt_img_var.grad.data.zero_()
        adv_ref_right_img_var.grad.data.zero_()
        bs,c,h,w= adv_tgt_img_grad.shape
        
        #pgdattack
        patch_var += args.lr*torch.sign(adv_tgt_img_grad)
        patch_var = torch.clamp(patch_var,-1,1)
        
        

        adv_tgt_img_var = torch.mul((1-mask_var), tgt_img_var) + torch.mul(mask_var, patch_var) 
        advpad = nn.ZeroPad2d(padding=(0,disparity,0,0))
        ref_ma = advpad(patch_var[:,:,:,disparity:])
        adv_ref_right_img_var = torch.mul((1-ref_mask_var), ref_right_img_var) + torch.mul(ref_mask_var, ref_ma)

        adv_tgt_img_var = torch.clamp(adv_tgt_img_var, -1, 1)

        adv_ref_right_img_var = torch.clamp(adv_ref_right_img_var, -1, 1)

        loss_scalar = loss.item()

        if args.log_terminal:
            logger.attack_bar.update(count)

        if count > args.max_count-1:
            break

    return adv_tgt_img_var, adv_ref_right_img_var, patch_var,loss_scalar

def validate_stereo_with_gt(patch, mask, patch_shape, val_loader, stereo_net,epoch, logger, output_writer,errors,error_names,dirs):
    global args
    batch_time = AverageMeter()
    errors = AverageMeter(i=len(error_names))
    

    stereo_net.eval()
    
    
            
    end = time.time()
    with torch.no_grad():
        for i, ( tgt_img, ref_img_right, disp_gt,_, fn) in enumerate(val_loader):
            tgt_img_var = Variable(tgt_img.cuda())
            
            ref_img_right_var = Variable(ref_img_right.cuda())
            
            disp_gt_var = Variable(disp_gt.unsqueeze(0).cuda())
            disp_gt_numpy = disp_gt[0].numpy()
            
            data_shape = tgt_img.cpu().numpy().shape
            
            stereo_fwd = stereo_net.forward(tgt_img_var, ref_img_right_var,mode_type='DM')


            
            
            if args.patch_type == 'circle':
                patch_full, mask_full, _, _, _, _ = circle_transform(patch, mask, patch.copy(), data_shape, patch_shape)
            elif args.patch_type == 'square' or 'checker':
                ref_patch, ref_mask, _, ref_rx, ref_ry = square_transform(patch, mask, patch.copy(), data_shape, patch_shape,patch_shape, norotate=args.norotate) #右边向里
                patch_full, mask_full, _, rx, ry = square_transform(patch, mask, patch.copy(), data_shape, patch_shape,patch_shape, norotate=args.norotate,fixed_loc=(ref_rx, ref_ry)) #左图向外
                
            
            
            patch_full, mask_full = torch.FloatTensor(patch_full), torch.FloatTensor(mask_full)

            patch_full, mask_full = patch_full.cuda(), mask_full.cuda()
            patch_var, mask_var = Variable(patch_full), Variable(mask_full)
            ####################
            
            ref_patch, ref_mask = torch.FloatTensor(ref_patch), torch.FloatTensor(ref_mask)
            
            ref_patch, ref_mask = ref_patch.cuda(), ref_mask.cuda()
            
            ref_patch_var, ref_mask_var = Variable(ref_patch), Variable(ref_mask)
            ######################

            adv_tgt_img_var = torch.mul((1-mask_var), tgt_img_var) + torch.mul(mask_var, patch_var)
            
            adv_ref_img_right_var = torch.mul((1-ref_mask_var), ref_img_right_var) + torch.mul(ref_mask_var, ref_patch_var)

            

            adv_tgt_img_var = torch.clamp(adv_tgt_img_var, -1, 1)
            
            adv_ref_img_right_var = torch.clamp(adv_ref_img_right_var, -1, 1)
            adv_stereo_fwd = stereo_net.forward(adv_tgt_img_var, adv_ref_img_right_var,mode_type='DM')
            #################存储了压缩的文件##########################
                    
            adv_stereo_fwdBL=torch.zeros_like(disp_gt_var)
            
            
            
        
            mask3pixel = ((disp_gt_var < 192) & (disp_gt_var > 0))
            
            bt, _, h_gt, w_gt = disp_gt_var.shape
            
            mask_var_res = F.interpolate(mask_var, size=(h_gt, w_gt), mode='bilinear')
            mask_var_res = (mask_var_res[:,0,:,:]).unsqueeze(1)
            disp_gt_var_adv = torch.mul((1-mask_var_res), disp_gt_var) + torch.mul(mask_var_res, adv_stereo_fwdBL)
            
            
            patch_disparity_var = torch.ones_like(disp_gt_var)*(-ref_rx+rx)
            stereo_true = torch.mul(disp_gt_var,1-mask_var_res)+torch.mul(patch_disparity_var,mask_var_res)
            stereo_true_np = stereo_true.data[0,0].cpu().numpy()
            
            peo_disparity_var = torch.ones_like(adv_stereo_fwd)*(-ref_rx+rx)
            peo_stereo_true = torch.mul(stereo_fwd,1-mask_var[:,0,:,:])+torch.mul(peo_disparity_var,mask_var[:,0,:,:])           
            peo_stereo_true_np = peo_stereo_true.data[0,0].cpu().numpy()
            bt, _, h_d, w_d = adv_stereo_fwd.shape
            disp_gt_var_full = F.interpolate(disp_gt_var,size=(h_d, w_d),mode='bilinear',align_corners=True)
            mask3pixel_full = ((disp_gt_var_full < 192) & (disp_gt_var_full > 0))
               
            
            
            epe = compute_epe(gt=disp_gt_var, pred=stereo_fwd,mask=mask3pixel) #这个flow-gt-var 还是彩色的
            D1_Metric = D1_metric(gt=disp_gt_var, pred=stereo_fwd,mask=mask3pixel)
            D1_Metric_adv = D1_metric(gt=stereo_true, pred=adv_stereo_fwd,mask=mask3pixel)
            adv_epe = compute_epe(gt=stereo_true, pred=adv_stereo_fwd,mask=mask3pixel)
            D1peo_Metric = D1_metric(gt=peo_stereo_true, pred=adv_stereo_fwd,mask=mask3pixel,mask_flag=False)
            adv_cos_sim = compute_cossim(stereo_true, adv_stereo_fwd)

            errors.update([epe, adv_epe, D1peo_Metric, adv_cos_sim,D1_Metric,D1_Metric_adv])
            
            
            
            val_tgt_image = tensor2array(tgt_img_var.data.cpu()[0]) 
            val_adv_tgt_image = tensor2array(adv_tgt_img_var.data.cpu()[0]) 
            val_tgt_image_cv2 = cv2.cvtColor(val_tgt_image ,cv2.COLOR_RGB2BGR)               
            val_adv_tgt_image_cv2 = cv2.cvtColor(val_adv_tgt_image ,cv2.COLOR_RGB2BGR)
            
            val_right_image = tensor2array(ref_img_right_var.data.cpu()[0]) 
            val_adv_right_image = tensor2array(adv_ref_img_right_var.data.cpu()[0])
            val_right_image_cv2 = cv2.cvtColor(val_right_image ,cv2.COLOR_RGB2BGR)             
            val_adv_right_image_cv2 = cv2.cvtColor(val_adv_right_image ,cv2.COLOR_RGB2BGR)  
            
            
            val_stereo_peo = tensor2array(stereo_fwd.data[0].cpu())/255.
            val_stereo_output = tensor2array(peo_stereo_true.data[0].cpu())/255.  
            val_adv_stereo_output = tensor2array(adv_stereo_fwd.data[0].cpu())/255. 
            diff_stereo_output = abs(adv_stereo_fwd-peo_stereo_true).data[0,0].cpu().numpy()
            Diff_D1_all = np.where(diff_stereo_output>=3)                            
            temp =np.ones_like(diff_stereo_output)*0.05
            temp[Diff_D1_all]=0.9
            val_Diff_stereo_output = cv2.applyColorMap((255*temp).astype(np.uint8), cv2.COLORMAP_JET)/255. 
            
            
            
            temp1 = np.hstack((val_adv_tgt_image_cv2,val_adv_right_image_cv2))  
            temp2 = np.hstack((val_stereo_output,val_adv_stereo_output))         
            temp3 = np.hstack((val_Diff_stereo_output,val_tgt_image_cv2))
            val_output_cv2 = (np.vstack((temp1,temp2,temp3))*255).astype(np.uint8)
            
            
            
            
            fn = os.path.join(dirs+'/', fn[-1])
            print("saving to", fn)
            cv2.imwrite(fn,val_output_cv2)

            if args.log_terminal:
                logger.valid_bar.update(i)

            batch_time.update(time.time() - end)
            end = time.time()


    return errors.avg, error_names


if __name__ == '__main__':
    import sys
    with open("experiment_recorder.md", "a") as f:
        f.write('\n python3 ' + ' '.join(sys.argv))
    main()
