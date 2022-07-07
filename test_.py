# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 16:46:16 2022

@author: Minyoung
"""

import os

import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from utils.plot import (
    Recorder,
    PlotManager,
    plot,
    )

from utils.utils import Cutout

from model.ae import Classifier, AutoEncoder

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# For Device
CUDA                =       torch.cuda.is_available()
DEVICE              =       torch.device('cuda' if CUDA else 'cpu')

    
def test(config,
         weights_file_name,
         model_type,
         use_pre_calculated_data= False,
         ):
    
    if use_pre_calculated_data:
        result = torch.load('test_result.pt')
        return result['gt'], result['preds']
        
    else:
    
        # --------------------------------------------------- MODEL -------------------------------------------------
        
        if model_type == 'classifier':
            model = Classifier().to(DEVICE)
        elif model_type == 'autoencoder':
            model = AutoEncoder().to(DEVICE)
            
                
        print(model)
        
        load_path = f'weights/{model_type}/{weights_file_name}'
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint['model_dict'])
      
        model.eval()

        # ------------------------------------------------- DATASET LOADER --------------------------------------------------
        test_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((80, 80)),
            # torchvision.transforms.CenterCrop((64, 64)),
            torchvision.transforms.ToTensor(),
            Cutout(32),
            ])   
        
        test_dset = torchvision.datasets.CIFAR10('../../data/CIFAR10/', train= False, transform= test_transform, download= True)
        # test_dset = torchvision.datasets.CIFAR100('../../data/CIFAR100/', train= False, transform= test_transform, download= True)
        test_loader = torch.utils.data.DataLoader(test_dset, batch_size= config.batch_size, shuffle= True, drop_last= True)

        # --------------------------------------------------- TRAIN -------------------------------------------------------    
        
        with torch.no_grad():
            
            loop = tqdm(test_loader)
            loop.set_description(f'{model_type.upper()} Testing')
    
            gt = []        
            preds = []     
            
            for batch_idx, (img, label) in enumerate(loop):
                            
                gt.append(label)
                
                img = img.to(DEVICE)
                label = label.to(DEVICE)
                label_one_hot = F.one_hot(label, config.num_classes).to(torch.float32)
                
                if model_type == 'classifier':
                    pred = model(img[:, 1])
                elif model_type == 'autoencoder':
                    pred, _ = model(img)
                            
                _, pred_indices = torch.max(pred.data, dim= -1)
                # acc_test = ((pred_indices == label).sum() / img.size(0)).item()
                
                preds.append(pred_indices)
                
            gt = torch.cat(gt, dim= 0)
            preds = torch.cat(preds, dim= 0).detach().cpu()
        
        torch.save({
            'gt' : gt,
            'preds' : preds,
            }, 'test_result.pt')
        
        return gt, preds
                 
        
def calculate_score(gt, preds, weighted= False):
       
    from sklearn.metrics import precision_score, recall_score, accuracy_score, average_precision_score, f1_score
    
    if weighted:
        a = accuracy_score(gt, preds,)
        b = precision_score(gt, preds, average= 'weighted')
        c = recall_score(gt, preds, average= 'weighted')
        
        print(f'Total Accuracy \t\t\t\t\t: \t {100 * a:.2f} %')
        print(f'Weighted Average of Precision \t: \t {100 * b:.2f} %')
        print(f'Weighted Average of Recall \t\t: \t {100 * c:.2f} %')
    
    else:
        a = accuracy_score(gt, preds,)
        b = precision_score(gt, preds, average= None)
        c = recall_score(gt, preds, average= None)
        
        b, b_ = torch.std_mean(torch.tensor(b), dim= 0)
        c, c_ = torch.std_mean(torch.tensor(c), dim= 0)
        
        print(f'Total Accuracy \t: \t {100 * a:.2f} %')
        print(f'Precision \t\t: \t {100 * b_:.2f} +- {100 * b:.2f} %')
        print(f'Recall \t\t\t: \t {100 * c_:.2f} +- {100 * c:.2f} %')        
        
def show_matrix(data1, data2, option= 'confusion matrix', **kwarg):
        
    label = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck',]
    
    if option == 'confusion matrix':
        '''
        data1 : ground truth
        data2 : prediction
        '''
        
        plt.figure()
        cf_matrix = confusion_matrix(data1, data2)

        ax = sns.heatmap(cf_matrix, annot=True, fmt= "d", xticklabels= label, yticklabels= label,
                           cmap= sns.light_palette("seagreen", as_cmap=True))
                          # cmap= sns.color_palette("Blues", as_cmap=True))
        ax.set_title('Confusion Matrix of Phase Recognition')
        ax.set_xlabel('Prediction', loc= 'right')
        ax.set_ylabel('Ground Truth', loc= 'bottom')
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
            


if __name__ == '__main__':
    
    import cfgs.test.cifar10 as config
    # import cfgs.test.cifar100 as config
    
    weights_file_name = 'ep_0001.pt'
    
    # model_type = 'classifier'
    model_type = 'autoencoder'
    
    gt, preds = test(config, weights_file_name, model_type, )
    calculate_score(gt, preds)
    show_matrix(gt, preds,)