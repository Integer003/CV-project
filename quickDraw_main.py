# 在 quickDraw 数据集上训练 VAE 模型

import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np
import os
from vae_quickDraw import VAE, Encoder, Decoder


# load data
path = 'quickDraw/full_numpy_bitmap_cat.npy'
data = np.load(path)
print(data.shape)
data = data/255.0
data = data.astype('float32')

np.random.shuffle(data)

# split the data
train_data = data[:100000]
test_data = data[100000:120000]


# Hyperparameters, 输入维度，隐藏层维度，z维度
input_dim = 784
hidden_dim = 400
latent_dim = 20



# vae.load_state_dict(torch.load('quickDraw/vae_quickDraw.pth'))

epochs = 201 #训练时期
batch_size = 1024 #每步训练样本数
learning_rate = 1e-4 #学习率
device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')#训练设备

model = VAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
modelname = 'quickDraw/vae_quickDraw_200.pth'
try:
    model.load_state_dict(torch.load(modelname))
    print('[INFO] Load Model complete')
except:
    pass


optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#损失函数
#交叉熵，衡量各个像素原始数据与重构数据的误差
loss_BCE = torch.nn.BCELoss(reduction = 'sum')
#均方误差可作为交叉熵替代使用.衡量各个像素原始数据与重构数据的误差
loss_MSE = torch.nn.MSELoss(reduction = 'sum')
#KL散度，衡量正态分布(mu,sigma)与正态分布(0,1)的差异，来源于公式计算
loss_KLD = lambda mu,sigma: -0.5 * torch.sum(1 + torch.log(sigma**2) - mu.pow(2) - sigma**2)

loss_history = {'train':[],'eval':[]}

def eval():
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for i in range(0, len(test_data), batch_size):
            x = torch.tensor(test_data[i:i+batch_size]).float().to(device)
            x = x.view(-1, input_dim)
            re_x, mu, sigma = model(x)
            # 重构误差
            loss_bce = loss_BCE(re_x, x)
            # KL散度
            loss_kld = loss_KLD(mu, sigma)
            # 总损失
            loss = loss_bce + loss_kld
            eval_loss += loss.item()
    print(f'Eval loss: {eval_loss/len(test_data)}')
    loss_history['eval'].append(eval_loss/len(test_data))

def visualize():    
    ### 采样并生成图片
    model.eval()
    with torch.no_grad():
        z = torch.randn(64, latent_dim).to(device)
        re_x = model.decoder(z)
        re_x = re_x.view(-1, 28, 28).cpu().numpy()
        fig, axes = plt.subplots(8, 8, figsize=(8, 8))
        for i in range(8):
            for j in range(8):
                axes[i, j].imshow(re_x[i*8+j], cmap='gray')
                axes[i, j].axis('off')
        plt.show()

def train(eval_interval=10, visualize_interval = 100, epochs=epochs):
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for i in range(0, len(train_data), batch_size):
            x = torch.tensor(train_data[i:i+batch_size]).float().to(device)
            x = x.view(-1, input_dim)
            re_x, mu, sigma = model(x)
            # 重构误差
            loss_bce = loss_BCE(re_x, x)
            # KL散度
            loss_kld = loss_KLD(mu, sigma)
            # 总损失
            loss = loss_bce + loss_kld
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f'Epoch {epoch}: Train loss: {train_loss/len(train_data)}')
        loss_history['train'].append(train_loss/len(train_data))
        if epoch % eval_interval == 0:
            torch.save(model.state_dict(), f'quickDraw/vae_quickDraw_{epoch}.pth')
            eval()
        if epoch % visualize_interval == 0:
            visualize()
    
train()