### 在CIFAR10数据集上训练VAE模型
###

import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F
from tqdm import tqdm
import os
from vae_cifar10_3l import Encoder, Decoder, VAE

# os.chdir(os.path.dirname(__file__))

#损失函数
#交叉熵，衡量各个像素原始数据与重构数据的误差
loss_BCE = torch.nn.BCELoss(reduction = 'sum')
#均方误差可作为交叉熵替代使用.衡量各个像素原始数据与重构数据的误差
loss_MSE = torch.nn.MSELoss(reduction = 'sum')
#KL散度，衡量正态分布(mu,sigma)与正态分布(0,1)的差异，来源于公式计算
loss_KLD = lambda mu,sigma: -0.5 * torch.sum(1 + torch.log(sigma**2) - mu.pow(2) - sigma**2)


'超参数及构造模型'
#模型参数
latent_size =1024 #压缩后的特征维度
hidden_size = 512 #encoder和decoder中间层的维度
input_size= output_size = 3*32*32 #原始图片和生成图片的维度

#训练参数
epochs = 50 #训练时期
batch_size = 128 #每步训练样本数
learning_rate = 1e-4 #学习率
device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')#训练设备

#确定模型，导入已训练模型（如有）
modelname = 'vae.pth'
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
try:
    model.load_state_dict(torch.load(modelname))
    print('[INFO] Load Model complete')
except:
    pass



'训练模型'
#CIFAR10 (数据会下载到py文件所在的data文件夹下)
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('/data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('/data', train=False, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=False)

# 可视化一个batch的数据
# imgs, lbls = next(iter(train_loader))
# plt.imshow(imgs[0].permute(1,2,0))
# plt.show()
# print(imgs.shape)
# print(lbls.shape)
# print(lbls[0])

#训练及测试
loss_history = {'train':[],'eval':[]}


def train():
    for epoch in range(epochs):   
        #训练
        model.train()
        #每个epoch重置损失，设置进度条
        train_loss = 0
        train_nsample = 0
        t = tqdm(train_loader,desc = f'[train]epoch:{epoch}')
        for imgs, lbls in t: #imgs:(bs,3,32,32) lbls:(bs)
            bs = imgs.shape[0]
            #获取数据
            imgs = imgs.to(device)
            #模型运算     
            re_imgs, mu, sigma = model(imgs)
            #计算损失
            loss_re = loss_BCE(re_imgs, imgs) # 重构与原始数据的差距(也可使用loss_MSE)
            loss_norm = loss_KLD(mu, sigma) # 正态分布(mu,sigma)与正态分布(0,1)的差距
            loss = loss_re + loss_norm
            #反向传播、参数优化，重置
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            #计算平均损失，设置进度条
            train_loss += loss.item()
            train_nsample += bs
            t.set_postfix({'loss':train_loss/train_nsample})
        #每个epoch记录总损失
        loss_history['train'].append(train_loss/train_nsample)

        #测试
        model.eval()
        #每个epoch重置损失，设置进度条
        test_loss = 0
        test_nsample = 0
        e = tqdm(test_loader,desc = f'[eval]epoch:{epoch}')
        for imgs, label in e:
            bs = imgs.shape[0]
            #获取数据
            imgs = imgs.to(device)
            #模型运算   
            re_imgs, mu, sigma = model(imgs)
            #计算损失
            loss_re = loss_BCE(re_imgs, imgs) 
            loss_norm = loss_KLD(mu, sigma) 
            loss = loss_re + loss_norm
            #计算平均损失，设置进度条
            test_loss += loss.item()
            test_nsample += bs
            e.set_postfix({'loss':test_loss/test_nsample})
        #每个epoch记录总损失    
        loss_history['eval'].append(test_loss/test_nsample)

        print(f'epoch:{epoch} train_loss:{train_loss/train_nsample} eval_loss:{test_loss/test_nsample}')

        
        if epoch%10==0:
            #展示效果   
            #按标准正态分布取样来自造数据
            sample = torch.randn(1,latent_size).to(device)
            #用decoder生成新数据
            gen = model.decoder(sample)[0].view(3,32,32)
            #将测试步骤中的真实数据、重构数据和上述生成的新数据绘图
            concat = torch.cat((imgs[0].view(3,32,32), re_imgs[0].view(3,32,32), gen), 2)
            show_concat = concat.cpu().detach().numpy().transpose(1,2,0)
            plt.imshow(show_concat)
            plt.show()

        # #显示每个epoch的loss变化
        # plt.plot(range(epoch+1),loss_history['train'])
        # plt.plot(range(epoch+1),loss_history['eval'])
        # plt.show()
        #存储模型
        torch.save(model.state_dict(),modelname)

# train()

'测试模型'
#展示效果
model.eval()
# samples = torch.randn(10,latent_size).to(device)
# gens = model.decoder(samples)
# gens = gens.view(-1,3,32,32).cpu().detach().numpy()
# fig,ax = plt.subplots(1,10,figsize=(20,2))
# for i in range(10):
#     ax[i].imshow(gens[i].transpose(1,2,0))
#     ax[i].axis('off')
# plt.show()

# pick from test_loader
imgs, lbls = next(iter(train_loader))
imgs = imgs.to(device)
re_imgs, mu, sigma = model(imgs)

# # show imgs
imgs = imgs.cpu().detach().numpy()
fig,ax = plt.subplots(3,10,figsize=(20,2))
for i in range(10):
    ax[0,i].imshow(imgs[i].transpose(1,2,0))
    ax[0,i].axis('off')
    ax[1,i].imshow(re_imgs[i].cpu().detach().numpy().transpose(1,2,0))
    ax[1,i].axis('off')
    # label
    ax[2,i].text(0.5,0.5,lbls[i].item(),fontsize=20)
    ax[2,i].axis('off')
plt.show()
print(lbls[:10])

# 0 airplane, 1 automobile, 2 bird, 3 cat, 4 deer, 5 dog, 6 frog, 7 horse, 8 ship, 9 truck
# find a airplane
# pick from test_loader
# imgs, lbls = next(iter(test_loader))
# img_airplane = imgs[lbls==0][1]
# latent_airplane = model.encoder(img_airplane.unsqueeze(0).to(device))[0]
# img_auto = imgs[lbls==1][0]
# latent_auto = model.encoder(img_auto.unsqueeze(0).to(device))[0]
# for i in range(11):
#     latent = latent_airplane*(1-i/10) + latent_auto*(i/10)
#     gen = model.decoder(latent)[0].view(3,32,32)
#     axes = plt.subplot(1,11,i+1)
#     axes.imshow(gen.cpu().detach().numpy().transpose(1,2,0))
#     axes.axis('off')
# plt.show()

# find a automobile