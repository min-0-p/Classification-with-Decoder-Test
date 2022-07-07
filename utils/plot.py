# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 22:18:18 2021

@author: MinYoung
"""


import matplotlib.pyplot as plt
import pandas as pd
import torch

class Recorder():
    def __init__(self, interval= 1):

        self.interval = interval
        self.lr = []
        self.data_dict = {}
        
    def to_csv(self, path):
        self.df.to_csv(path, index= False)
            
    def plot_lr(self,):    
        plt.figure()
        
        x = range(1, (len(self.lr) + 1))
        plt.plot(x, self.lr, label= 'learning rate')
            
        plt.xlabel(f'Iteration')
        plt.ylabel('Value')
        plt.title('Training Schedule')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot(self, *keys):  
        self.df = pd.DataFrame(self.data_dict)
        plt.figure()
        
        for key in keys:
            column = self.df[key]
            data = []
            for i in range(0, len(column), self.interval):
                data.append(column.values[i:i+self.interval].mean())
            
            x = range(1, (len(data) + 1))
            plt.plot(x, data, label= column.name)
            
        plt.xlabel(f'Epochs')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()
            
    def __getitem__(self, key):
        return self.data_dict[key]
        
        
class PlotManager():
    def __init__(self, path):

        self.interval = 100
        self.df = pd.read_csv(path,)
        
    def __getitem__(self, key):
        return self.df[key]

    def plot(self, *series):    
        plt.figure()
        
        for column in series:
            data = []
            for i in range(0, len(column), self.interval):
                data.append(column.values[i:i+self.interval].mean())
            
            x = range(1, (len(data) + 1))
            plt.plot(x, data, label= column.name)
            
        plt.xlabel(f'Iteration (x{self.interval})')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()
        
def plot(tensor_3d):
    
    imgs = []
    for i in range(tensor_3d.shape[0]):
        imgs.append(tensor_3d[i])
    imgs = torch.cat(imgs, dim= -2)
    
    plt.imshow(imgs.detach().cpu().permute(1,2,0))
    # plt.grid(64)
    plt.show()  

if __name__ == '__main__':
   pass