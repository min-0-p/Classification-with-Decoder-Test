# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 13:03:23 2021

@author: MinYoung
"""
import os, sys

import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from matplotlib import pyplot as plt
from tqdm import tqdm

class Cutout(nn.Module):
    def __init__(self, size):
        self.size = size
        super(Cutout, self).__init__()
        
    def forward(self, tensor_img):
        device = 'cpu' if tensor_img.get_device() == -1 else tensor_img.get_device()
        original = tensor_img.clone()
            
        C, H, W = tensor_img.size()    
        y, x = torch.randint(0, H, (1,), device= device), torch.randint(0, W, (1,), device= device)
        
        # (y_min, y_max), (x_min, x_max)
        (y_min, y_max), (x_min, x_max) = (torch.max(torch.zeros_like(y, dtype= torch.int64), y - self.size // 2), torch.min(y + self.size // 2, H * torch.ones_like(y, dtype= torch.int64))),\
                                         (torch.max(torch.zeros_like(x, dtype= torch.int64), x - self.size // 2), torch.min(x + self.size // 2, W * torch.ones_like(x, dtype= torch.int64)))
        tensor_img[:, y_min:y_max, x_min:x_max] = 0.5
        # out_tensor.requires_grad = True
        
        return torch.stack([tensor_img, original], dim= 0)

def plot(tensor_3d):
    
    imgs = []
    for i in range(tensor_3d.shape[0]):
        imgs.append(tensor_3d[i])
    imgs = torch.cat(imgs, dim= -2)
    
    plt.imshow(imgs.detach().cpu().permute(1,2,0))
    # plt.grid(64)
    plt.show()


if __name__ == '__main__':
    
    img_dirs = [f for f in os.listdir() if f.endswith(".jpg")]
    imgs = torch.stack([torchvision.io.read_image(img_dir) / 255.0 for img_dir in img_dirs], dim= 0)
    
    print(imgs.size())
    
    cutout = Cutout(size= 130)
    for img in imgs:
        imgs = cutout(img, )
        plot(imgs)