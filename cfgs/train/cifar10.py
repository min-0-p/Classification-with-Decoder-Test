# -*- coding: utf-8 -*-1563
"""
Created on Thu May 26 18:18:22 2022

@author: Minyoung
"""

# H Params for Training
batch_size = 64
dataset_size = 50000
num_classes = 10

num_epochs = 100

lr = 1E-4
weight_decay = 1E-4
eta_min = 1E-6

restart = 4
has_zero_remainder = (dataset_size % batch_size) == 0
T_0 = (dataset_size // batch_size) * (num_epochs // restart) if has_zero_remainder else ((dataset_size // batch_size) + 1 ) * (num_epochs // restart)
