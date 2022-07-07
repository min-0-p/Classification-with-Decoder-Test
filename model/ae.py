# -*- coding: utf-8 -*-
"""
Created on Wed May 25 13:46:20 2022

@author: Minyoung
"""

import torch, torchvision
from torch import nn
from torch.nn import functional as F

from torchsummary import summary

# [in_channels, out_channels, downsample,  ]
encoder_cfgs = [    
    [ 64,  64, False, ],
    [ 64, 128,  True, ],
    
    [128, 128, False, ],
    [128, 256,  True, ],
    
    [256, 256, False, ],
    [256, 512,  True, ],
    
    [512, 512, False, ],    
    ]

# [in_channels, out_channels, upsample,  ]
decoder_cfgs = [    
    [512, 256,  True, ],
    [256, 256, False, ],
    
    [256, 128,  True, ],
    [128, 128, False, ],
    
    [128,  64,  True, ],
    [ 64,  64, False, ],
    ]

class EncoderBlock(nn.Module):
    def __init__(self, cfg,):
        super(EncoderBlock, self).__init__()
        
        in_channels, out_channels, self.have_downsample = cfg
        kernel_size = 3
        stride = 2 if self.have_downsample else 1
        padding = kernel_size // 2
        
        layers = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,),
            nn.BatchNorm2d(out_channels,),
            nn.ReLU(inplace= True),
            nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding,),
            nn.BatchNorm2d(out_channels),
            ])
        
        
        self.block = nn.Sequential(*layers)
        
        if self.have_downsample: 
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.BatchNorm2d(out_channels,), 
                nn.ReLU(),
                )
        
    def forward(self, x):
        out = x + self.block(x) if not self.have_downsample else self.downsample(x) + self.block(x)
        return out


class Encoder(nn.Module):
    def __init__(self, cfgs, is_part_of_autoencoder= False):
        super(Encoder, self).__init__()
        self.is_part_of_autoencoder = is_part_of_autoencoder
        
        layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, cfgs[0][0], 3, 1, 1,),
                nn.BatchNorm2d(cfgs[0][0]),
                nn.ReLU(inplace= True),            
                )
            ])
        
        for cfg in cfgs:
            layers.append(EncoderBlock(cfg))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        
        if self.is_part_of_autoencoder:
            N, C, H, W = x.size()
            
            features = []
            for i, layer in enumerate(self.model):
                x = layer(x)
                if i % 2 == 0: features.append(x[:N//2])
            features.reverse()
            return x, features
        
        else:            
            return self.model(x)
    


class DecoderBlock(nn.Module):
    def __init__(self, cfg,):
        super(DecoderBlock, self).__init__()
        
        in_channels, out_channels, self.have_upsample = cfg
        kernel_size = 3
        stride = 1
        padding = kernel_size // 2
                
        
        layers = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,),
            nn.InstanceNorm2d(out_channels,),
            nn.ReLU(inplace= True),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding,),
            nn.InstanceNorm2d(out_channels,),
            ])
        
        if self.have_upsample: 
            layers.insert(0, nn.UpsamplingNearest2d(scale_factor= 2))
            self.upsample = nn.Sequential(
                nn.UpsamplingNearest2d(scale_factor= 2),
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,),
                # nn.InstanceNorm2d(out_channels,),
                nn.ReLU(inplace= True,),
                )
        
        self.block = nn.Sequential(*layers)
        
    def forward(self, x):
        out = x + self.block(x) if not self.have_upsample else self.upsample(x) + self.block(x)
        return out
        

class Decoder(nn.Module):
    def __init__(self, cfgs,):
        super(Decoder, self).__init__()
        
        layers = nn.ModuleList()        
        for cfg in cfgs:
            layers.append(DecoderBlock(cfg))
        
        layers.append(
            nn.Sequential(
                nn.Conv2d(cfgs[-1][1], 3, 3, 1, 1,),
                nn.Sigmoid(),            
                )
            )
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x, features):
        
        N, C, H, W = x.size()
        
        for i, layer in enumerate(self.model):
            if i % 2 == 0: x += features[i//2]
            x = layer(x)
            
        return x
        

class Classifier(nn.Module):
    def __init__(self, num_classes= 10):
        super(Classifier, self).__init__()
        
        self.features = Encoder(encoder_cfgs)
        self.fc = nn.Conv2d(encoder_cfgs[-1][1], num_classes, 1)
        
    def forward(self, x):
        out = self.features(x)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = self.fc(out).squeeze()
        
        return out
        
        
class AutoEncoder(nn.Module):
    
    def __init__(self, num_classes= 10):
        super(AutoEncoder, self).__init__()
        
        self.encoder = Encoder(encoder_cfgs, is_part_of_autoencoder= True)
        self.decoder = Decoder(decoder_cfgs)
        self.fc = nn.Conv2d(encoder_cfgs[-1][1], num_classes, 1)
        
    def forward(self, x):
        '''
        x.size() = [N, 2, C, H, W]
        x[:, 0] -> Cutout images
        x[:, 1] -> Original images
        '''
        x = x.transpose(0, 1)
        _, N, C, H, W = x.size()
        x = x.reshape(-1, C, H, W)
        
        encoder_out, features = self.encoder(x)
        decoder_out = self.decoder(encoder_out[:N], features)
                
        probs = self.fc(F.adaptive_avg_pool2d(encoder_out[N:], (1, 1))).reshape(N, -1)
        
        # import sys
        # print(probs.size())
        # print(decoder_out.size())
        # sys.exit()
        
        return probs, decoder_out


if __name__ == '__main__':
    
    x = torch.randn(4, 3, 224, 224).to('cuda')
    y = torch.randn(4, 2, 3, 224, 224).to('cuda')
    
    c = Classifier().to('cuda')
    ae = AutoEncoder().to('cuda')
    
    out1 = c(x)
    out2, imgs = ae(y)
    
    print(out1.size())
    print(out2.size())