import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.measure import label
import numpy as np
from pytorch_msssim import MS_SSIM, ms_ssim, SSIM, ssim


class MS_SSIM_Loss(MS_SSIM):
    def forward(self, img1, img2):
        img1 = (img1 + 1) / 2
        img2 = (img2 + 1) / 2
        return 5*(1 - super(MS_SSIM_Loss, self).forward(img1, img2))


class SSIM_Loss(SSIM):
    def forward(self, img1, img2):
        img1 = (img1 + 1) / 2
        img2 = (img2 + 1) / 2
        return 5*(1 - super(SSIM_Loss, self).forward(img1, img2))
def weight_self_pro_softmax_mse_loss(input_logits, target_logits,entropy):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """

    target_logits = target_logits .view(target_logits .size(0), target_logits .size(1), -1)
    target_logits = target_logits.transpose(1, 2)  # [N, HW, C]
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=2)
    target_softmax = F.softmax(target_logits, dim=2)
    mse_loss = (input_softmax.detach()-target_softmax)**2

    return mse_loss

def weight_cross_pro_softmax_mse_loss(weight,input_logits, target_logits): ##target_logits==classfier input_logit = cross_prototype
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    weight = F.softmax(weight,dim=2)
    target_logits = target_logits .view(target_logits .size(0), target_logits .size(1), -1)
    target_logits = target_logits.transpose(1, 2)  # [N, HW, C]
    assert input_logits.size() == target_logits.size()
    #input_softmax = F.softmax(input_logits, dim=2)
    target_softmax = F.softmax(target_logits, dim=2)
    mse_loss = (input_logits.detach()-target_softmax)**2
    mse_loss = weight.detach()*mse_loss
    return mse_loss



def softmax_kl_loss(input_logits, target_logits, sigmoid=False):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    if sigmoid:
        input_log_softmax = torch.log(torch.sigmoid(input_logits))
        target_softmax = torch.sigmoid(target_logits)
    else:
        input_log_softmax = F.log_softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)

    # return F.kl_div(input_log_softmax, target_softmax)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='mean')
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])
    return kl_div
def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss
def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    #target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax-target_logits)**2
    return mse_loss
def softmax_mae_loss_EGV(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    #target_softmax = F.softmax(target_logits, dim=1)

    mae_loss = torch.abs(input_softmax-target_logits)
    return mae_loss
def softmax_mse_loss_VGE(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    target_softmax = F.softmax(target_logits, dim=1)
    #target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_logits-target_softmax.detach())**2
    return mse_loss
def softmax_mae_loss_VGE(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    target_softmax = F.softmax(target_logits, dim=1)
    #target_softmax = F.softmax(target_logits, dim=1)

    mae_loss = torch.abs(input_logits-target_softmax.detach())
    return mae_loss
def softmax_mae_loss_DiceCE(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    #target_softmax = F.softmax(target_logits, dim=1)
    #target_softmax = F.softmax(target_logits, dim=1)

    mae_loss = torch.abs(input_logits-target_logits.detach())
    return mae_loss
def dce_eviloss(p, alpha, c, global_step, annealing_step):
    #evidence = F.softplus(prob)
    # L_dice =  TDice(alpha,p,criterion_dl)
    alpha = alpha.view(alpha.size(0), alpha.size(1), -1)  # [N, C, HW]
    alpha = alpha.transpose(1, 2)  # [N, HW, C]
    alpha = alpha.contiguous().view(-1, alpha.size(2))
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    label = label.view(-1, c)
    # digama loss
    a = torch.tensor(([1,10]))
    L_ace = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha))*a.cuda(), dim=1, keepdim=True)
    L_ace = torch.mean(L_ace.squeeze())
    #print(L_ace.shape)
    # log loss
    # labelK = label * (torch.log(S) -  torch.log(alpha))
    # L_ace = torch.sum(label * (torch.log(S) -  torch.log(alpha)), dim=1, keepdim=True)

    annealing_coef = min(1, global_step / annealing_step)*0.4
    alp = E * (1 - label) + 1
    L_KL = annealing_coef * KL(alp, c)
    L_KL = torch.mean(L_KL.squeeze())
    return L_ace  + L_KL
def dce_eviloss_2d(p, alpha, c, global_step, annealing_step):
    #evidence = F.softplus(prob)
    # L_dice =  TDice(alpha,p,criterion_dl)
    alpha = alpha.view(alpha.size(0), alpha.size(1), -1)  # [N, C, HW]
    alpha = alpha.transpose(1, 2)  # [N, HW, C]
    alpha = alpha.contiguous().view(-1, alpha.size(2))
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p.long(), num_classes=c)
    label = label.view(-1, c)
    # digama loss
    a = torch.tensor(([1,5,10,5]))
    L_ace = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha))*a.cuda(), dim=1, keepdim=True)
    L_ace = torch.mean(L_ace.squeeze())
    #print(L_ace.shape)
    # log loss
    # labelK = label * (torch.log(S) -  torch.log(alpha))
    # L_ace = torch.sum(label * (torch.log(S) -  torch.log(alpha)), dim=1, keepdim=True)

    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    L_KL = annealing_coef * KL(alp, c)
    L_KL = torch.mean(L_KL.squeeze())
    return (L_ace  + L_KL)
def L_KL(label, alpha, c, global_step, annealing_step):
    #evidence = F.softplus(prob)
    # L_dice =  TDice(alpha,p,criterion_dl)
    alpha = alpha.view(alpha.size(0), alpha.size(1), -1)  # [N, C, HW]
    alpha = alpha.transpose(1, 2)  # [N, HW, C]
    alpha = alpha.contiguous().view(-1, alpha.size(2))
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(label, num_classes=c)
    label = label.view(-1, c)
    # digama loss
    # log loss
    # labelK = label * (torch.log(S) -  torch.log(alpha))
    # L_ace = torch.sum(label * (torch.log(S) -  torch.log(alpha)), dim=1, keepdim=True)

    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    L_KL = annealing_coef * KL(alp, c)
    L_KL = torch.mean(L_KL.squeeze())
    return  L_KL
def unlabelplainloss(unlabelpred, alpha, c):
    #evidence = F.softplus(prob)
    # L_dice =  TDice(alpha,p,criterion_dl)
    alpha = alpha.view(alpha.size(0), alpha.size(1), -1)  # [N, C, HW]
    alpha = alpha.transpose(1, 2)  # [N, HW, C]
    alpha = alpha.contiguous().view(-1, alpha.size(2))
    S = torch.sum(alpha, dim=1, keepdim=True)
    W = 1-c/S
    P = alpha/S
    unlabelpred = unlabelpred.view(unlabelpred.size(0), unlabelpred.size(1), -1)
    unlabelpred = unlabelpred.transpose(1, 2)  # [N, HW, C]
    unlabelpred = unlabelpred.contiguous().view(-1, unlabelpred.size(2))
    L_con_1 = torch.sum(softmax_mse_loss(unlabelpred,P.detach())*W,dim=1)/2
    L_con_1 = torch.mean(L_con_1)
    return L_con_1
def unlabelplainmaeloss(unlabelpred, alpha, c):
    #evidence = F.softplus(prob)
    # L_dice =  TDice(alpha,p,criterion_dl)
    alpha = alpha.view(alpha.size(0), alpha.size(1), -1)  # [N, C, HW]
    alpha = alpha.transpose(1, 2)  # [N, HW, C]
    alpha = alpha.contiguous().view(-1, alpha.size(2))
    S = torch.sum(alpha, dim=1, keepdim=True)
    W = 1-c/S
    P = alpha/S
    unlabelpred = unlabelpred.view(unlabelpred.size(0), unlabelpred.size(1), -1)
    unlabelpred = unlabelpred.transpose(1, 2)  # [N, HW, C]
    unlabelpred = unlabelpred.contiguous().view(-1, unlabelpred.size(2))
    L_con_1 = torch.sum(softmax_mae_loss_EGV(unlabelpred,P.detach())*W,dim=1)/2
    L_con_1 = torch.mean(L_con_1)
    return L_con_1
def unlabelcrossclassiferloss(unlabelpred, alpha,c):
    #evidence = F.softplus(prob)
    # L_dice =  TDice(alpha,p,criterion_dl)
    alpha = alpha.view(alpha.size(0), alpha.size(1), -1)  # [N, C, HW]
    alpha = alpha.transpose(1, 2)  # [N, HW, C]
    alpha = alpha.contiguous().view(-1, alpha.size(2))
    S = torch.sum(alpha, dim=1, keepdim=True)
    W = 1-c/S
    P = alpha/S
    unlabelpred = unlabelpred.view(unlabelpred.size(0), unlabelpred.size(1), -1)
    unlabelpred = unlabelpred.transpose(1, 2)  # [N, HW, C]
    unlabelpred = unlabelpred.contiguous().view(-1, unlabelpred.size(2))
    L_con_1 = torch.sum(softmax_mse_loss(unlabelpred,P.detach())*W.detach(),dim=1)/2
    L_con_2 = torch.sum(softmax_mse_loss_VGE(P,unlabelpred.detach())*W.detach(),dim=1)/2
    L_con = torch.mean(L_con_1+L_con_2)
    
    return L_con
def unlabelcrossclassifermaeloss(unlabelpred, alpha,c):
    #evidence = F.softplus(prob)
    # L_dice =  TDice(alpha,p,criterion_dl)
    alpha = alpha.view(alpha.size(0), alpha.size(1), -1)  # [N, C, HW]
    alpha = alpha.transpose(1, 2)  # [N, HW, C]
    alpha = alpha.contiguous().view(-1, alpha.size(2))
    S = torch.sum(alpha, dim=1, keepdim=True)
    W = 1-c/S
    P = alpha/S
    unlabelpred = unlabelpred.view(unlabelpred.size(0), unlabelpred.size(1), -1)
    unlabelpred = unlabelpred.transpose(1, 2)  # [N, HW, C]
    unlabelpred = unlabelpred.contiguous().view(-1, unlabelpred.size(2))
    L_con_1 = torch.sum(softmax_mae_loss_EGV(unlabelpred,P.detach())*W.detach(),dim=1)/2
    L_con_2 = torch.sum(softmax_mae_loss_VGE(P,unlabelpred.detach())*W.detach(),dim=1)/2
    L_con = torch.mean(L_con_1+L_con_2)
    return L_con
def no_meanmae_loss(input1, input2):
    return(torch.abs(input1 - input2))
def DiceCEmaeconloss(unlabelpred, alpha,c):
    #evidence = F.softplus(prob)
    # L_dice =  TDice(alpha,p,criterion_dl)
    alpha = alpha.view(alpha.size(0), alpha.size(1), -1)  # [N, C, HW]
    alpha = alpha.transpose(1, 2)  # [N, HW, C]
    alpha = alpha.contiguous().view(-1, alpha.size(2))
    S1 = torch.sum(alpha, dim=1, keepdim=True)
    W1 = 1-c/S1
    P1 = alpha/S1
    unlabelpred = unlabelpred.view(unlabelpred.size(0), unlabelpred.size(1), -1)
    unlabelpred = unlabelpred.transpose(1, 2)  # [N, HW, C]
    unlabelpred = unlabelpred.contiguous().view(-1, unlabelpred.size(2))
    S2 = torch.sum(unlabelpred, dim=1, keepdim=True)
    W2 = 1-c/S2
    P2 = unlabelpred/S2
    L_con_1 = torch.sum(mse_loss(P2,P1.detach())*W1.detach(),dim=1)/2
    L_con_2 = torch.sum(mse_loss(P1,P2.detach())*W2.detach(),dim=1)/2
    L_con = torch.mean(L_con_1+L_con_2)#*5
    return L_con
def Binary_dice_loss(predictive, target, ep=1e-8):
    intersection = 2 * torch.sum(predictive * target) + ep
    union = torch.sum(predictive) + torch.sum(target) + ep
    loss = 1 - intersection / union
    return loss

def kl_loss(inputs, targets, ep=1e-8):
    kl_loss=nn.KLDivLoss(reduction='mean')
    consist_loss = kl_loss(torch.log(inputs+ep), targets)
    return consist_loss

def soft_ce_loss(inputs, target, ep=1e-8):
    logprobs = torch.log(inputs+ep)
    return  torch.mean(-(target[:,0,...]*logprobs[:,0,...]+target[:,1,...]*logprobs[:,1,...]))

def mse_loss(input1, input2):
    return torch.mean((input1 - input2)**2)
def mae_loss(input1, input2):
    return torch.mean(torch.abs(input1 - input2))
def get_cut_mask(probs, thres=0.5, nms=0):
    masks = (probs >= thres).type(torch.int64)
    masks = masks[:, 1, :, :].contiguous()
    if nms == 1:
        masks = LargestCC_pancreas(masks)
    return masks


def LargestCC_pancreas(segmentation):
    N = segmentation.shape[0]
    batch_list = []
    for n in range(N):
        n_prob = segmentation[n].detach().cpu().numpy()
        labels = label(n_prob)
        if labels.max() != 0:
            largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        else:
            largestCC = n_prob
        batch_list.append(largestCC)

    return torch.Tensor(batch_list).cuda()

def one_resize_mse_loss(input1, input2):
    input1 = input1.view(input1.size(0), input1.size(1), -1)  # [N, C, HW]
    input1 = input1.transpose(1, 2)  # [N, HW, C]
    input1 = input1.contiguous().view(-1, input1.size(2))
    return torch.mean((input1 - input2.detach())**2)
def uncertainty_mse_loss(input1, input2,uncertainty):
    return torch.mean(((input1 - input2)**2)*(1-uncertainty.unsqueeze(1).detach())) 
def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-10
        intersection = torch.sum(score * target)
        union = torch.sum(score * score) + torch.sum(target * target) + smooth
        loss = 1 - intersection / union
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes
CE = nn.CrossEntropyLoss()
Dice= DiceLoss(2)
def one_resize_CE_loss(input1, input2):
    patch_size = input1.size()
    input2 = input2.view(patch_size[0],-1,patch_size[1])
    input2 = input2.transpose(1,2)
    input2 = input2.reshape(patch_size[0],patch_size[1],patch_size[2],patch_size[3],patch_size[4])
    input2 = get_cut_mask(input2,0.5,1)
    return CE(input1, input2.long())#+Dice(input1/torch.sum(input1,dim=1,keepdim=True),input2.unsqueeze(1))
class DiceLoss_evi(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss_evi, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-10
        intersection = torch.sum(score * target)
        union = torch.sum(score) + torch.sum(target) + smooth
        loss = 1 - intersection / union
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        #print(inputs.shape,target.shape)
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes
class evidentDiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(evidentDiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        print(output_tensor.shape)
        return output_tensor.float()

    def _dice_loss(self, alpha, target):
        S = torch.sum(alpha, dim=1, keepdim=True)
        p = alpha/S
        target = target.float()
        smooth = 1e-10
        intersection = torch.sum(p * target)
        sDiceDen = torch.sum(p * p) + torch.sum(target * target) + smooth
        varfenzi = alpha*(S-alpha)
        varfenmu = S*S*(S+1)
        var = torch.sum(varfenzi/varfenmu)
        union = sDiceDen + var
        sumi = intersection / union
        return sumi

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        #class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            #class_wise_dice.append(dice.item())
            loss += dice * weight[i]
        loss = 1- (loss/self.n_classes *2)
        return loss
def softmax_mae_loss(alpha1, alpha2,c):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    S1 = torch.sum(alpha1, dim=1, keepdim=True)
    W1 = 1 - c / S1
    P1= alpha1 / S1
    S2 = torch.sum(alpha2, dim=1, keepdim=True)
    W2 = 1 - c/S2
    P2 = alpha2 / S2
    loss1 = torch.sum(torch.abs(P1 - P2.detach())*W2) / (alpha1.shape[0] * alpha1.shape[1] * alpha1.shape[2] * alpha1.shape[3] * alpha1.shape[4])
    loss2 = torch.sum(torch.abs(P2 - P1.detach())*W1) / (alpha1.shape[0] * alpha1.shape[1] * alpha1.shape[2] * alpha1.shape[3] * alpha1.shape[4])
    loss = loss1 + loss2
    #input_softmax = F.softmax(input_logits, dim=1)
    #target_softmax = F.softmax(target_logits, dim=1)
    return loss