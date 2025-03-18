## CIFAR10 的 三层 Conv 的 VAE

import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F
from tqdm import tqdm
import os



'模型结构'
class Encoder(torch.nn.Module):
    #编码器，将input_size维度数据压缩为latent_size维度的mu和sigma
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 4, stride=2, padding=1)  # 32,16,16
        self.conv2 = torch.nn.Conv2d(64, 128, 4, stride=2, padding=1) # 64,8,8
        self.conv3 = torch.nn.Conv2d(128, 512, 4, stride=2, padding=1) # 128,4,4
        self.mu = torch.nn.Linear(512*4*4, 1024)
        self.sigma = torch.nn.Linear(512*4*4, 1024)
    def forward(self, x):# x: bs,input_size
        x = F.relu(self.conv1(x)) #->bs,32,16,16
        x = F.relu(self.conv2(x)) #->bs,64,8,8
        x = F.relu(self.conv3(x))
        x = x.view(-1,512*4*4) #->bs,64*8*8
        mu = self.mu(x) #->bs,128
        sigma = self.sigma(x) #->bs,128
        return mu,sigma

class Decoder(torch.nn.Module):
    #解码器，将latent_size维度的数据转换为output_size维度的数据
    def __init__(self):
        super(Decoder, self).__init__()
        self.linear1 = torch.nn.Linear(1024, 128*4*4)
        self.conv1 = torch.nn.ConvTranspose2d(128, 256, 4, stride=2, padding=1)
        self.conv2 = torch.nn.ConvTranspose2d(256, 64, 4, stride=2, padding=1)
        self.conv3 = torch.nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1)
    def forward(self, x): # x:bs,latent_size
        x = F.relu(self.linear1(x)) #->bs,64*8*8
        x = x.view(-1,128,4,4) #->bs,64,8,8
        x = F.relu(self.conv1(x)) #->bs,32,16,16
        x = F.relu(self.conv2(x)) #->bs,16,32,32
        x = torch.sigmoid(self.conv3(x))
        return x

class VAE(torch.nn.Module):
    #将编码器解码器组合
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    def forward(self, x): #x: bs,input_size
        # 压缩，获取mu和sigma
        mu,sigma = self.encoder(x) #mu,sigma: bs,latent_size
        # 采样，获取采样数据
        eps = torch.randn_like(sigma)  #eps: bs,latent_size
        z = mu + eps*sigma  #z: bs,latent_size
        # 重构，根据采样数据获取重构数据
        re_x = self.decoder(z) # re_x: bs,output_size
        return re_x,mu,sigma