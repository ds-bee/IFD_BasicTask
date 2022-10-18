#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn.functional as F


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    global mmd
    """
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val) / len(kernel_val)


def pc_guassian_kernel_for_one_class(source, target, pl_sour, pl_targ, device, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    pcmmd (for every class×)
    """
    n_samples = int(source.size()[0])+int(target.size()[0])
    batch = n_samples//2
    # create weight mat:

    pl_sour = F.softmax(pl_sour,1)
    _, ps = torch.max(pl_sour, 1)
    pl_targ = F.softmax(pl_targ,1)
    _, pt = torch.max(pl_targ, 1)
    w_mat = torch.zeros(n_samples, n_samples)

    # XX:
    for i in range(batch):
        for j in range(batch):
            w_mat[j][i] = pl_sour[i][ps[i]]*pl_sour[j][ps[i]]
    # XY
    for i in range(batch):
        for j in range(batch, n_samples):
            w_mat[j][i] = pl_sour[i][ps[i]]*pl_targ[j-batch][ps[i]]
    # YX
    for i in range(batch, n_samples):
        for j in range(batch):
            w_mat[j][i] = pl_targ[i-batch][pt[i-batch]]*pl_sour[j][pt[i-batch]]
    # YY
    for i in range(batch, n_samples):
        for j in range(batch, n_samples):
            w_mat[j][i] = pl_targ[i-batch][pt[i-batch]]*\
            pl_targ[j-batch][pt[i-batch]]

    # calculate kernel mat
    
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)

    # PCMMD:
    for i in range(n_samples):
        for j in range(n_samples):
            # L2_distance[i][j] = float(L2_distance[i][j])*w_mat[i][j]
            L2_distance[i][j] = float(L2_distance[i][j])
    w_mat = w_mat.to(device)
    L2_distance = L2_distance.to(device)

    L2_distance = L2_distance * w_mat

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val) #/ len(kernel_val)


def calculate_pl_mat(pl_sour, pl_targ):
    total = torch.cat([pl_sour, pl_targ], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    w_mat = total0 * total1
    return w_mat


def pc_guassian_kernel_for_all_class(source, target, pl_sour, pl_targ, device, class_c, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    pcmmd (for every class)
    """
    n_samples = int(source.size()[0])+int(target.size()[0])
    batch = n_samples//2

    pl_sour = F.softmax(pl_sour, 1)
    pl_targ = F.softmax(pl_targ, 1)
    w_mat = calculate_pl_mat(pl_sour, pl_targ)

    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_d = ((total0-total1)**2).sum(2)

    # PCMMD:
    w_mat = w_mat.to(device)
    L2_distance = torch.zeros(L2_d.shape).to(device)
    for i in range(len(w_mat[0][0])):
        # L2_distance = L2_distance + L2_d * w_mat[:, :, i]
        L2_distance = L2_distance + class_c[i] * L2_d * w_mat[:, :, i]
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)


def mmd_rbf_accelerate(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    loss = 0
    for i in range(batch_size):
        s1, s2 = i, (i+1) % batch_size
        t1, t2 = s1+batch_size, s2+batch_size
        loss += kernels[s1, s2] + kernels[t1, t2]
        loss -= kernels[s1, t2] + kernels[s2, t1]
    return loss / float(batch_size)


def mmd_rbf_noaccelerate(source, target, pl_sour, pl_targ, class_c, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size = int(source.size()[0])
    kernels_pc = pc_guassian_kernel_for_all_class(source, target, pl_sour, pl_targ, device, class_c,
                                                  kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    kernels_global = guassian_kernel(source, target, 
                                     kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    kernels = 0.5*(kernels_global.to(device)) + kernels_pc
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return abs(loss)


def mmd_rbf_test(source, target, pl_sour, pl_targ, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size = int(source.size()[0])
    # kernels_pc = pc_guassian_kernel_for_all_class(source, target, pl_sour, pl_targ, device,
    #             kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    kernels_global = guassian_kernel(source, target, 
                kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    # kernels = 0.5*kernels_global + 0.5*kernels_pc
    kernels = (kernels_global.to(device))
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss


if __name__ == "__main__":
    
    import torch
    import numpy as np
    num = 16
    li = []
    for i in range(1, num+1):
        lis = []
        for j in range(i, i+512):
            lis.append(j/512.)
        li.append(lis)
    print("li.length=", len(li))
    source = torch.tensor(li)
    target = torch.tensor(li)
    sour = torch.randn(16, 5)
    targ = torch.randn(16, 5)
    mmd_rbf_noaccelerate(source, target, sour, targ)


