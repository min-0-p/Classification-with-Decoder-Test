# -*- coding: utf-8 -*-
"""
Created on Wed May 25 20:55:36 2022

@author: Minyoung
"""

from train import train
from test_ import test, calculate_score, show_matrix

# mode = 'train'
mode = 'test'
mode = 'train & test'

import cfgs.train.cifar10 as cfg
# import cfgs.train.cifar100 as cfg

# model_type = 'classifier'
model_type = 'autoencoder'

validation = True
# validation = False


weights_file_name = 'ep_0100.pt'

if __name__ == '__main__':   
    
    
    
    if 'train' in mode:
    
        train(cfg, model_type, validation= validation)
    
    if 'test' in mode :
    
        gt, preds = test(cfg, weights_file_name, model_type, )
        calculate_score(gt, preds)
        show_matrix(gt, preds,)