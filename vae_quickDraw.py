# quickDraw数据集的纯线性层的VAE模型

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
    def __init__(self, input_dim,hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.mu = torch.nn.Linear(hidden_dim, latent_dim)
        self.sigma = torch.nn.Linear(hidden_dim, latent_dim)
    def forward(self, x):# x: bs,input_size
        x = F.relu(self.fc1(x)) 
        mu = self.mu(x) 
        sigma = self.sigma(x) 
        return mu,sigma

class Decoder(torch.nn.Module):
    #解码器，将latent_size维度的数据转换为output_size维度的数据
    def __init__(self, input_dim,hidden_dim, latent_dim):
        super(Decoder, self).__init__()
        self.fc1 = torch.nn.Linear(latent_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, input_dim)
    def forward(self, x): # x:bs,latent_size
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

class VAE(torch.nn.Module):
    #将编码器解码器组合
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim,hidden_dim=hidden_dim, latent_dim=latent_dim)
        self.decoder = Decoder(input_dim,hidden_dim=hidden_dim, latent_dim=latent_dim)
    def forward(self, x): #x: bs,input_size
        # 压缩，获取mu和sigma
        mu,sigma = self.encoder(x) #mu,sigma: bs,latent_size
        # 采样，获取采样数据
        eps = torch.randn_like(sigma)  #eps: bs,latent_size
        z = mu + eps*sigma  #z: bs,latent_size
        # 重构，根据采样数据获取重构数据
        re_x = self.decoder(z) # re_x: bs,output_size
        return re_x,mu,sigma