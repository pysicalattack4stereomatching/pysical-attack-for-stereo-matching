import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
epsilon = 1e-8
def compute_epe(gt, pred,mask):
    _, _, h_pred, w_pred = pred.size()
    _, _, h_gt, w_gt = gt.size()
    pred = F.interpolate(pred,size=(h_gt,w_gt),mode='bilinear',align_corners=True)
    
  
    avg_epe = F.l1_loss(gt[mask], pred[mask], size_average=True)


    if type(avg_epe) == Variable: avg_epe = avg_epe.data

    return avg_epe.item()

def compute_cossim(gt, pred):
    _, _, h_pred, w_pred = pred.size()
    bs, nc, h_gt, w_gt = gt.size()
    #u_gt, v_gt = gt[:,0,:,:], gt[:,1,:,:]
    pred = F.interpolate(pred,size=(h_gt,w_gt),mode='bilinear',align_corners=True)
    #u_pred = pred[:,0,:,:] * (w_gt/w_pred)
    #v_pred = pred[:,1,:,:] * (h_gt/h_pred)
    similarity = F.cosine_similarity(gt[:,:2], pred)
    if nc == 3:
        valid = gt[:,2,:,:]
        similarity = similarity * valid
        avg_sim = similarity.sum()/(valid.sum() + epsilon)
    else:
        avg_sim = similarity.sum()/(bs*h_gt*w_gt)


    if type(avg_sim) == Variable: avg_sim = avg_sim.data

    return avg_sim.item()

def multiscale_cossim(gt, pred):
    assert(len(gt)==len(pred))
    loss = 0
    for (_gt, _pred) in zip(gt, pred):
        loss +=  - nn.functional.cosine_similarity(_gt, _pred).mean()

    return loss

def D1_metric( gt,pred, mask,mask_flag=True):
    _, _, h_pred, w_pred = pred.size()
    _, _, h_gt, w_gt = gt.size()
    pred = F.interpolate(pred,size=(h_gt,w_gt),mode='bilinear',align_corners=True)   
    if mask_flag==False:
        pred, gt = pred, gt
    else :
        pred, gt = pred[mask], gt[mask]   
    E = torch.abs(gt - pred)
    err_mask = (E > 3) & (E / gt.abs() > 0.05)
    return torch.mean(err_mask.float())

def get_loss( pred, target,mode='epe'):
    if mode == 'mse':
        return torch.mean((pred - target)**2)
    else:
        diff_squared = (pred-target)**2
        if len(diff_squared.size()) == 3:
            # here, dim=0 is the 2-dimension (u and v direction of flow [2,M,N]) , which needs to be added BEFORE taking the square root. To get the length of a flow vector, we need to do sqrt(u_ij^2 + v_ij^2)
            epe = torch.mean(torch.sum(diff_squared, dim=0).sqrt())
        elif len(diff_squared.size()) == 4:
            # here, dim=0 is the 2-dimension (u and v direction of flow [b,2,M,N]) , which needs to be added BEFORE taking the square root. To get the length of a flow vector, we need to do sqrt(u_ij^2 + v_ij^2)
            epe = torch.mean(torch.sum(diff_squared, dim=1).sqrt())
        elif len(diff_squared.size()) >= 5 and len(diff_squared.size()) <= 6:
            epe = torch.mean(torch.sum(diff_squared).sqrt())
        else:
            raise ValueError("The flow tensors for which the EPE should be computed do not have a valid number of dimensions (either [b,2,M,N] or [2,M,N]). Here: " + str(flow1.size()) + " and " + str(flow1.size()))
    return epe