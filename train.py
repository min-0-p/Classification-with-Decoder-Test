# -*- coding: utf-8 -*-
"""
Created on Wed May 25 20:55:29 2022

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

# For Device
CUDA                =       torch.cuda.is_available()
DEVICE              =       torch.device('cuda' if CUDA else 'cpu')

def train(
        config,
        model_type,
        weights_file_name= None,
        validation= True,
        use_cutout= False,
        ):
        
    # --------------------------------------------------- MODEL -------------------------------------------------

    if model_type == 'classifier' : model = Classifier().to(DEVICE)
    elif model_type == 'autoencoder': model = AutoEncoder().to(DEVICE)
            
    print(model)
    
    # ------------------------------------------------ OPTIMIZER -----------------------------------------------------
    
    # Optimizers & LearningRate Schedulers
        
    optimizer = torch.optim.AdamW(model.parameters(), lr= config.lr, weight_decay= config.weight_decay)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = config.T_max, eta_min= config.eta_min)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = config.T_0, eta_min= config.eta_min)
    
    
    if weights_file_name == None:
        step = 1 # For iteration saving
        recorder = Recorder()
        
    else:
        load_path = f'weights/{model_type}/{weights_file_name}'
        checkpoint = torch.load(load_path)
        step = checkpoint['step']
        recorder = checkpoint['recorder']
        model.load_state_dict(checkpoint['model_dict'])
        optimizer.load_state_dict(checkpoint['optim_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_dict'])
    
    model.train()
    
    os.makedirs(f'weights/{model_type}/', exist_ok=True)
    os.makedirs(f'record/{model_type}/', exist_ok=True)
    
    # ------------------------------------------------- DATASET LOADER --------------------------------------------------
    
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((80, 80)),
        torchvision.transforms.RandomCrop((64, 64)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        Cutout(32),
        ])
    train_dset = torchvision.datasets.CIFAR10('../../data/CIFAR10/', train= True, transform= transform, download= True)
    # train_dset = torchvision.datasets.CIFAR100('../../data/CIFAR100/', train= True, transform= transform, download= True)
    train_loader = torch.utils.data.DataLoader(train_dset, batch_size= config.batch_size, shuffle= True)
    
    if validation:    
        val_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((80, 80)),
            # torchvision.transforms.CenterCrop((64, 64)),
            torchvision.transforms.ToTensor(),
            Cutout(32),
            ])        
        validation_dset = torchvision.datasets.CIFAR10('../../data/CIFAR10/', train= False, transform= val_transform, download= True)
        # validation_dset = torchvision.datasets.CIFAR100('../../data/CIFAR100/', train= False, transform= val_transform, download= True)
        val_loader = torch.utils.data.DataLoader(validation_dset, batch_size= config.batch_size, shuffle= True, drop_last= True)
        
    
    # ------------------------------------------------ CRITERION -----------------------------------------------------
    criterion_class = nn.BCEWithLogitsLoss()
    criterion_inpainting = nn.MSELoss()
    # --------------------------------------------------- TRAIN -------------------------------------------------------    
    
    recorder.data_dict['loss_train'] = []
    recorder.data_dict['loss_class'] = []
    if model_type == 'autoencoder': recorder.data_dict['loss_inpainting'] = []
    recorder.data_dict['acc_train'] = []
    if validation: recorder.data_dict['loss_val'] = []; recorder.data_dict['acc_val'] = []
    
    for epoch in range(config.num_epochs):
        
        loop = tqdm(train_loader)
        loop.set_description(f'Epoch [{epoch + 1:3d}/{config.num_epochs}]      Train Step')
    
        count = 0
        total_correct = 0
        total_class_loss = 0.0
        if model_type == 'autoencoder': total_inpainting_loss = 0.0
        
        for batch_idx, (imgs, label) in enumerate(loop):
            
            step += 1
            
            imgs = imgs.to(DEVICE) 
            label = label.to(DEVICE)
            label_one_hot = F.one_hot(label, config.num_classes).to(torch.float32)
            
            if model_type == 'classifier':
                
                pred = model(imgs[:, 0] if use_cutout else imgs[:, 1])
                loss_class = criterion_class(pred, label_one_hot)
                loss = loss_class
                
                count += label.size(0)
                total_class_loss += loss_class.item() * label.size(0)                           
                postfix = f'Class Loss = {loss_class.item():.4f}, ' 
                
            elif model_type == 'autoencoder':
                
                pred, inpainted_imgs = model(imgs)
                loss_class = criterion_class(pred, label_one_hot)
                loss_inpainting = criterion_inpainting(inpainted_imgs, imgs[:, 1].data)
                loss = loss_class + loss_inpainting
                
                count += label.size(0)
                total_class_loss += loss_class.item() * label.size(0)  
                total_inpainting_loss += loss_inpainting.item() * label.size(0)     
                postfix = f'Class Loss = {loss_class.item():.4f}, '        
                postfix += f'Inpainting Loss = {loss_inpainting.item():.4f}, ' 
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            _, pred_indices = torch.max(pred.data, dim= -1)
            correct = (pred_indices == label).sum().item()
            total_correct += correct
            acc_train = (correct / label.size(0))
            
            postfix += f'Train Accuracy = {acc_train * 100:.2f}%, '     
            loop.set_postfix_str(postfix)
                        
            recorder.lr.append(scheduler.get_last_lr()[0])
            scheduler.step()
            # if step < config.T_max : scheduler.step()
            
            if batch_idx % 100 == 0:
                if model_type == 'classifier':
                    plot(torch.cat([imgs[:4, 0], imgs[:4, 1]], dim= -1))
                elif model_type == 'autoencoder':
                    plot(torch.cat([imgs[:4, 0], inpainted_imgs[:4], imgs[:4, 1]], dim= -1))
                
        if model_type == 'classifier':
            total_loss = total_class_loss / count  
        elif model_type == 'autoencoder':
            total_loss = (total_class_loss + total_inpainting_loss) / count
                       
        recorder.data_dict['loss_train'].append(total_loss)
        recorder.data_dict['loss_class'].append(total_class_loss / count)
        if model_type == 'autoencoder': recorder.data_dict['loss_inpainting'].append(total_inpainting_loss / count)
        recorder.data_dict['acc_train'].append(total_correct / count)
 
        recorder.plot_lr()
        
        if validation: 
            criterion = (criterion_class, criterion_inpainting)
            loss_val, acc_val = validate(val_loader, model, criterion, model_type, epoch, config)
            recorder.data_dict['loss_val'].append(loss_val); recorder.data_dict['acc_val'].append(acc_val)
            
            if model_type == 'autoencoder':
                recorder.plot('loss_class', 'loss_inpainting')
            recorder.plot('loss_train', 'loss_val')
            recorder.plot('acc_train', 'acc_val')
                        
        else:
            if model_type == 'autoencoder':
                recorder.plot('loss_class', 'loss_inpainting')
            recorder.plot('acc_train')
            
        # Save File
        torch.save({
            'model_dict'                : model.state_dict(),
            'optim_dict'                : optimizer.state_dict(),
            'scheduler_dict'            : scheduler.state_dict(),
            'step'                      : step,
            'recorder'                  : recorder,
            }, f'weights/{model_type}/ep_{epoch + 1:04d}.pt',)
        recorder.to_csv(f'record/{model_type}/ep_{epoch:04d}.csv')
        
def validate(loader,
             model,
             criterion,
             model_type,
             epoch,
             config,
             ):
    
    criterion_class, criterion_inpainting = criterion    
    model.eval()

    count = 0
    total_correct = 0
    total_class_loss = 0.0
    if model_type == 'autoencoder': total_inpainting_loss = 0.0
        
    with torch.no_grad():

        loop = tqdm(loader)
        loop.set_description(f'Epoch [{epoch + 1:3d}/{config.num_epochs}] Validation Step')
    
        for batch_idx, (imgs, label) in enumerate(loop):
                        
            imgs = imgs.to(DEVICE)
            label = label.to(DEVICE)
            label_one_hot = F.one_hot(label, config.num_classes).to(torch.float32)

            if model_type == 'classifier':
                
                pred = model(imgs[:, 1])
                loss_class = criterion_class(pred, label_one_hot)
                         
                count += label.size(0)
                total_class_loss += loss_class.item() * label.size(0)          
                postfix = f'Class Loss = {loss_class.item():.4f}, ' 
                
            elif model_type == 'autoencoder':
                
                pred, inpainted_imgs = model(imgs)
                loss_class = criterion_class(pred, label_one_hot)
                loss_inpainting = criterion_inpainting(inpainted_imgs, imgs[:, 1].data)
                
                count += label.size(0)
                total_class_loss += loss_class.item() * label.size(0)  
                total_inpainting_loss += loss_inpainting.item() * label.size(0)     
                postfix = f'Class Loss = {loss_class.item():.4f}, '        
                postfix += f'Inpainting Loss = {loss_inpainting.item():.4f}, ' 
                            
            _, pred_indices = torch.max(pred.data, dim= -1)
            correct = (pred_indices == label).sum().item()
            total_correct += correct
            acc_val = (correct / label.size(0))
            
            postfix += f'Validation Accuracy = {acc_val * 100:.2f}%, '     
            loop.set_postfix_str(postfix) 
            
            if batch_idx % 100 == 0: 
                
                if model_type == 'classifier':
                    plot(torch.cat([imgs[:4, 0], imgs[:4, 1]], dim= -1))
                elif model_type == 'autoencoder':
                    plot(torch.cat([imgs[:4, 0], inpainted_imgs[:4], imgs[:4, 1]], dim= -1))
 
    if model_type == 'classifier':
        loss_val = total_class_loss / count  
    elif model_type == 'autoencoder':
        loss_val = (total_class_loss + total_inpainting_loss) / count
    
    model.train()
    return loss_val, acc_val

if __name__ == '__main__':
    
    import cfgs.train.cifar10 as cfg
    # import cfgs.train.cifar100 as cfg
    
    model_type = 'classifier'
    model_type = 'autoencoder'
    
    # train_classifier(cfg, validation= False)
    # train_autoencoder(cfg, validation= False)
    train(cfg, model_type, validation= True)
