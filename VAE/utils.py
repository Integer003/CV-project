from torchvision.datasets import MNIST
from torchvision import transforms
import torch
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split


def save_model(save_path,
               model,
               optimizer=None,
               replay_buffer=None,
               discriminator=None,
               optimizer_d=None):
    save_dict = {'model': model.state_dict()}
    if optimizer is not None:
        save_dict['optimizer'] = optimizer.state_dict()
    if replay_buffer is not None:
        save_dict['replay_buffer'] = replay_buffer
    if discriminator is not None:
        save_dict['discriminator'] = discriminator
    if optimizer_d is not None:
        save_dict['optimizer_d'] = optimizer_d
    torch.save(save_dict, save_path)


def load_model(load_path):
    checkpoint = torch.load(load_path)
    return (
        checkpoint['model'],
        checkpoint.get('optimizer', None),
        checkpoint.get('replay_buffer', None),
        checkpoint.get('discriminator', None),
        checkpoint.get('optimizer_d', None),
    )


def rescale(x, lo, hi):
    """Rescale a tensor to [lo,hi]."""
    assert (lo < hi), f"[rescale] lo={lo} must be smaller than hi={hi}"
    old_width = torch.max(x) - torch.min(x)
    old_center = torch.min(x) + (old_width / 2.)
    new_width = float(hi - lo)
    new_center = lo + (new_width / 2.)
    # shift everything back to zero:
    x = x - old_center
    # rescale to correct width:
    x = x * (new_width / old_width)
    # shift everything to the new center:
    x = x + new_center
    # return:
    return x


def corruption(x, type_='ebm', noise_scale=0.3):
    assert type_ in ['ebm']
    # mask=1 if the pixel is visible
    mask = torch.zeros_like(x)
    if type_ == 'ebm':
        # Corrupt the rows 0, 2, 4, ....
        mask[..., torch.arange(0, mask.shape[-2], step=2), :] = 1
    broken_data = x * mask + (1 - mask) * noise_scale * torch.randn_like(x)
    broken_data = torch.clip(broken_data, 1e-4, 1 - 1e-4)
    return broken_data, mask


transform = transforms.Compose([
    # convert PIL image to tensor:
    transforms.ToTensor(),
    # add uniform noise:
    transforms.Lambda(lambda x: (x + torch.rand_like(x).div_(256.))),
    # rescale to [0.0001, 0.9999]:
    transforms.Lambda(lambda x: rescale(x, 0.0001, 0.9999)),
])

# image tensor range [0, 1]
train_set = MNIST(
    root="./data",
    download=True,
    transform=transform,
    train=True,
)

val_set = MNIST(
    root="./data",
    download=True,
    transform=transform,
    train=False,
)

print(f"train_set size={len(train_set)}")
print(f"val_set size={len(val_set)}")

class MockResidualBlock(nn.Module):
    """ A simplified residual block used in RealNVP.

    Image feature shape is assumed to be unchanged.
    """

    def __init__(self,
                 c_in,
                 c_out,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 groups=1,
                 output_activation=True):
        super(MockResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c_in,
                      c_out,
                      kernel_size,
                      stride,
                      padding,
                      groups=groups,
                      bias=False), nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out,
                      c_out,
                      kernel_size,
                      stride,
                      padding,
                      groups=groups,
                      bias=False), nn.BatchNorm2d(c_out))
        self.skip_connection = nn.Identity()
        if c_in != c_out:
            self.skip_connection = nn.Sequential(nn.Conv2d(c_in, c_out, 1, 1),
                                                 nn.BatchNorm2d(c_out))

        self.output_activation = output_activation

    def forward(self, x):
        if self.output_activation:
            return F.relu(self.conv(x) + self.skip_connection(x), inplace=True)
        else:
            return self.conv(x) + self.skip_connection(x)


def hello():
    print('Good luck!')

def get_augmented_data(with_rot=False):
    if with_rot:
        augmented_imgs = torch.load('augmented_imgs_rot.pt')
        augmented_labels = torch.load('augmented_labels_rot.pt')
    else:
        augmented_imgs = torch.load('augmented_imgs.pt')
        augmented_labels = torch.load('augmented_labels.pt')

    print(f"augmented_imgs size={augmented_imgs.shape}") # (60000, 1, 28, 28)
    print(f"augmented_labels size={augmented_labels.shape}")  # (60000, 4)
    dataset = TensorDataset(augmented_imgs, augmented_labels)

    # 拆分数据集为训练集和验证集
    # 假设我们按照 80% 训练集，20% 验证集的比例拆分
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    augmented_train_set, augmented_val_set = random_split(dataset, [train_size, val_size])
    return augmented_train_set, augmented_val_set

# # show the first 10 images and labels
# import matplotlib.pyplot as plt
# import numpy as np
# def show_images(images, labels):
#     fig, axes = plt.subplots(2, 5, figsize=(10, 4))
#     for i in range(10):
#         ax = axes[i // 5, i % 5]
#         ax.imshow(images[i].squeeze().cpu().numpy(), cmap='gray')
#         ax.set_title(f"Label: {labels[i].cpu().numpy()}")
#         ax.axis('off')
#     plt.show()
# show_images(augmented_imgs, augmented_labels)



# 使用 DataLoader 包装数据集
# train_loader = DataLoader(augmented_train_set, batch_size=64, shuffle=True)
# val_loader = DataLoader(augmented_val_set, batch_size=64, shuffle=False)

# # 打印 DataLoader 的信息
# print(f"Train DataLoader size: {len(train_loader)} batches")
# print(f"Validation DataLoader size: {len(val_loader)} batches")

# # 示例：迭代 DataLoader
# for batch_idx, (data, target) in enumerate(train_loader):
#     print(f"Batch {batch_idx}: data shape={data.shape}, target shape={target.shape}")
#     # 在这里可以添加训练代码
#     break