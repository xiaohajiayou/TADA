import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-10
    # 将31本来和为0，通过entropy变成数值
    entropy = -input_ * torch.log2(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def im(outputs_test_raw, gent=True):
    # 16*31
    epsilon = 1e-10

    outputs_test  = outputs_test_raw
    # 将16*31的输入中的每个31进行概率初始化
    softmax_out = nn.Softmax(dim=1)(outputs_test)
    # 得到16*1，然后再求平均值
    entropy_loss = torch.mean(entropy(softmax_out))
    if gent:
        msoftmax = softmax_out.mean(dim=0)
        # 进行entropy，但是sum不一样
        gentropy_loss = torch.sum(-msoftmax * torch.log2(msoftmax + epsilon))
        entropy_loss -= gentropy_loss
    im_loss = entropy_loss * 1.0
    return im_loss
def advLoss(source, target, device):

    sourceLabel = torch.ones(len(source))
    targetLabel = torch.zeros(len(target))
    Loss = nn.BCELoss()
    if device == 'cuda':
        Loss = Loss.cuda()
        sourceLabel, targetLabel = sourceLabel.cuda(), targetLabel.cuda()
    #print("sd={}\ntd={}".format(source, target))
    loss = Loss(source, sourceLabel) + Loss(target, targetLabel)
    return loss*0.5
def entropy_advLoss(source, target, device):
    eps = 1e-10
    ad_out = torch.cat((source,target),dim=0)
    ad_out = ad_out.cpu().detach()
    entropyc = - ad_out * torch.log2(ad_out + eps) - (1.0 - ad_out) * torch.log2(1.0 - ad_out + eps)
    trans_ability = torch.mean(entropyc)
    return trans_ability


if __name__ == "__main__":
    pass