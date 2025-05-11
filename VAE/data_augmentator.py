from torchvision.datasets import MNIST
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F 
from utils import train_set, val_set
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_bbox(imgs, threshold=0.1):
    # imgs: bsz, (1), h, w
    """Get the bounding box of the non-zero pixels in a 2D image."""
    imgs = imgs.squeeze(1)  # Remove the channel dimension
    non_zero_coords = (imgs > threshold).to(torch.float32)  # Create a mask of non-zero pixels
    
    bsz = imgs.shape[0]
    
    # 计算每个图像的非零像素的最小和最大 x, y 坐标
    # print(non_zero_coords.any(dim=2))
    # print(non_zero_coords.any(dim=2).flip(dims=[1]))
    min_x = torch.argmax(non_zero_coords.any(dim=1).to(torch.float32), dim=1)
    max_x = imgs.shape[2] - torch.argmax(non_zero_coords.any(dim=1).flip(dims=[1]).to(torch.float32), dim=1) - 1
    min_y = torch.argmax(non_zero_coords.any(dim=2).to(torch.float32), dim=1)
    max_y = imgs.shape[1] - torch.argmax(non_zero_coords.any(dim=2).flip(dims=[1]).to(torch.float32), dim=1) - 1
    
    # 组合成 (bsz, 4) 的张量
    bboxes = torch.stack([min_x, max_x, min_y, max_y], dim=1)

    # print(bboxes)
    
    return min_x, max_x, min_y, max_y

def get_center(img):
    min_x, max_x, min_y, max_y = get_bbox(img)
    if min_x is None:
        return None, None
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    # print(f"min_x={min_x}, max_x={max_x}, min_y={min_y}, max_y={max_y}")
    # print(f"center_x={center_x}, center_y={center_y}")
    return center_x, center_y

def get_thickness(imgs, threadhold1=0.5, threadhold2=0.8*9):
    """Get the thickness"""
    imgs = imgs.squeeze(1)  # Remove the channel dimension
    thick_pixels = (imgs >= threadhold1)
    kernel = torch.tensor([[1, 1, 1],
                          [1, 1, 1],
                          [1, 1, 1]], dtype=torch.float32).to(imgs.device)
    inside_pixels = thick_pixels.float().clone()
    inside_pixels = nn.functional.conv2d(inside_pixels.unsqueeze(1), kernel.unsqueeze(0).unsqueeze(0), padding=1).squeeze(1)
    inside_pixels = (inside_pixels > threadhold2).float()
    inside_pix_cnt = (inside_pixels * thick_pixels).sum(dim=1).sum(dim=1)

    return inside_pix_cnt / (thick_pixels.sum(dim=1).sum(dim=1) + 1e-6)  # Avoid division by zero


def get_thickness_label(imgs):
    thickness_bars = [(0.05, 0.27), (0.35, 0.43), (0.5, 0.65)]
    thickness_bars = torch.tensor(thickness_bars, dtype=torch.float32).to(imgs.device)
    thickness = get_thickness(imgs)
    thickness_types = torch.zeros_like(thickness, dtype=torch.long)
    # for those thickness not in any bars, set to 3. o/w set to 0, 1, 2
    for i in range(thickness_bars.shape[0]):
        mask = (thickness >= thickness_bars[i][0]) & (thickness <= thickness_bars[i][1])
        thickness_types[mask] = 4-i
    thickness_types=4-thickness_types
    # print(f"thickness={thickness}, thickness_types={thickness_types}")
    return thickness_types

def mv_center_by_pos(imgs, new_center_x, new_center_y):
    # imgs: bsz, (1), h, w
    # new_center_x, new_center_y: (bsz, )
    imgs = imgs.squeeze(1)  # Remove the channel dimension
    # 获取当前图像的中心
    center_x, center_y = get_center(imgs) # (bsz, )

    # 计算平移量
    # print(f"center_x={center_x}, center_y={center_y}, new_center_x={new_center_x}, new_center_y={new_center_y}")
    translation_x = (new_center_x - center_x).type(torch.float32)  # (bsz, )
    translation_y = (new_center_y - center_y).type(torch.float32)  # (bsz, )
    # print(f"translation_x={translation_x}, translation_y={translation_y}")

    # 创建一个新的图像张量，大小与原图像相同
    new_imgs = torch.zeros_like(imgs)

    bsz, h, w = imgs.shape

    gridY = torch.linspace(-1, 1, steps = h).view(1, -1, 1, 1).expand(bsz, h, w, 1)
    gridX = torch.linspace(-1, 1, steps = w).view(1, 1, -1, 1).expand(bsz, h, w, 1)
    flow_grid = torch.cat((gridX, gridY), dim=3).type(torch.float32).to(imgs.device)
    # print(f"flow_grid.shape={flow_grid.shape}, translation_x.shape={translation_x.shape}, translation_y.shape={translation_y.shape}")


    # [TODO]: why * 2.5 ?
    flow_grid[:, :, :, 0] -= 2.5*translation_x.unsqueeze(1).unsqueeze(1) / (w - 1) 
    flow_grid[:, :, :, 1] -= 2.5*translation_y.unsqueeze(1).unsqueeze(1) / (h - 1) 

    new_imgs = F.grid_sample(imgs.unsqueeze(1), flow_grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    new_imgs = harden_img(new_imgs)
    new_imgs = new_imgs.squeeze(1).clamp(0, 1)  # Remove the channel dimension and clamp to [0, 1]

    return new_imgs

def mv_center(imgs, center_types):
    """Move the center of the image batch to the new centers."""
    center_types = torch.tensor(center_types, dtype=torch.long)
    assert torch.all((center_types >= 0) & (center_types <= 8)), f"center_types={center_types} has values not in [0...8]"
    new_imgs = imgs.clone()
    new_imgs = new_imgs.squeeze(1)
    center_list = [(10, 10), (10, 14), (10, 18),
                   (14, 10), (14, 14), (14, 18),
                   (18, 10), (18, 14), (18, 18)]
    center_list = torch.tensor(center_list, dtype=torch.float32).to(imgs.device)
    new_center_x = center_list[center_types][:, 0]
    new_center_y = center_list[center_types][:, 1]
    # print(f"new_center_x={new_center_x}, new_center_y={new_center_y}") # This is good
    return mv_center_by_pos(new_imgs, new_center_x, new_center_y)


def scale_img(input_imgs, size_type):
    size_type = torch.tensor(size_type, dtype=torch.long)
    assert torch.all((size_type >= 0) & (size_type <= 2)), f"size_type={size_type} has values not in [0...2]"
    sizes = [5.5, 9, 12]
    sizes = torch.tensor(sizes, dtype=torch.float32).to(input_imgs.device)
    new_sizes = sizes[size_type]
    # print(f"new_sizes={new_sizes}")
    imgs = input_imgs.clone()
    imgs = imgs.squeeze(1)
    ori_center_x, ori_center_y = get_center(imgs)
    # print(f"ori_center_x={ori_center_x}, ori_center_y={ori_center_y}")

    centersX= torch.tensor([14.]).repeat(imgs.shape[0]).to(imgs.device)
    centersY = torch.tensor([14.]).repeat(imgs.shape[0]).to(imgs.device)
    # print(f"centers.shape={centers.shape}, imgs.shape={imgs.shape}")
    imgs = mv_center_by_pos(imgs, centersX, centersY)

    min_x, max_x, min_y, max_y = get_bbox(imgs)

    
    ori_size_x = (max_x-min_x) / 2
    ori_size_y = (max_y-min_y) / 2
    
    scale_x = new_sizes / ori_size_x
    scale_y = new_sizes / ori_size_y

    scales = torch.stack([scale_x, scale_y], dim=1)
    scales = torch.min(scales, dim=1)[0]
    scale_x = scale_y = scales

    # print(f"ori_size_x={ori_size_x}, ori_size_y={ori_size_y}, scale_x={scale_x}, scale_y={scale_y}")
    
    # scale the image to the new size

    bsz, h, w = imgs.shape

    translation_x = torch.arange(w, dtype=torch.float32).view(1, 1, -1).expand(bsz, h, w).to(imgs.device)
    # print(f"translation_x.shape={translation_x}, scale_x.shape={scale_x.shape}")
    translation_x = (translation_x-14) /(scale_x.unsqueeze(1).unsqueeze(1)) + 14 # (bsz, h, w)
    # translation_y = (14 - ori_center_y) * (scale_y)
    # print(f"translation_x={translation_x}")

    translation_y = torch.arange(h, dtype=torch.float32).view(1, -1, 1).expand(bsz, h, w).to(imgs.device)
    translation_y = (translation_y-14) / (scale_y.unsqueeze(1).unsqueeze(1)) + 14 # (bsz, h, w)

    

    gridY = torch.linspace(-1, 1, steps = h).view(1, -1, 1, 1).expand(bsz, h, w, 1).to(imgs.device) \
        / (scale_y).to(imgs.device).unsqueeze(1).unsqueeze(1).unsqueeze(1)
    gridX = torch.linspace(-1, 1, steps = w).view(1, 1, -1, 1).expand(bsz, h, w, 1).to(imgs.device) \
        / (scale_x).to(imgs.device).unsqueeze(1).unsqueeze(1).unsqueeze(1)
    flow_grid = torch.cat((gridX, gridY), dim=3).type(torch.float32).to(imgs.device)

    # for those abs >1, set to 0
    masks = torch.abs(flow_grid) > 1
    img_masks = (torch.abs(flow_grid[...,0]) > 1) | (torch.abs(flow_grid[...,1]) > 1)
    img_masks = img_masks.float().unsqueeze(1)  # Add a channel dimension
    flow_grid = flow_grid * (1 - masks.float())
    # flow_grid = torch.where(torch.abs(flow_grid) > 1, torch.tensor(0.0).to(imgs.device), flow_grid)

    # print(flow_grid.shape, translation_x.shape, translation_y.shape)

    # flow_grid[:, :, :, 0] += translation_x / (w - 1)
    # flow_grid[:, :, :, 1] += translation_y / (h - 1)

    flow_grid = flow_grid.to(imgs.device)

    
    rescale_imgs = F.grid_sample(imgs.unsqueeze(1), flow_grid, mode='bilinear', padding_mode='zeros', align_corners=False)


    rescale_imgs = rescale_imgs*(1 - img_masks.float())  # Remove the channel dimension
    rescale_imgs = harden_img(rescale_imgs)

    new_img = mv_center_by_pos(rescale_imgs, ori_center_x, ori_center_y)
    
    return new_img

def rotate_img(imgs, angle_type):
    # imgs: bsz, (1), h, w
    # angle_type: bsz, [0, 1, 2, 3]
    angle_type = torch.tensor(angle_type, dtype=torch.long)
    assert torch.all((angle_type >= 0) & (angle_type <= 3)), f"angle_type={angle_type} has values not in [0...3]"
    angles = [0, 90, 180, 270]
    angles = torch.tensor(angles, dtype=torch.float32).to(imgs.device)
    new_angles = angles[angle_type]
    # print(f"new_angles={new_angles}")
    imgs = imgs.squeeze(1)  # Remove the channel dimension
    
    ori_center_x, ori_center_y = get_center(imgs)
    # print(f"ori_center_x={ori_center_x}, ori_center_y={ori_center_y}")

    centersX= torch.tensor([14.]).repeat(imgs.shape[0]).to(imgs.device)
    centersY = torch.tensor([14.]).repeat(imgs.shape[0]).to(imgs.device)
    # print(f"centers.shape={centers.shape}, imgs.shape={imgs.shape}")
    imgs = mv_center_by_pos(imgs, centersX, centersY)
    
    bsz, h, w = imgs.shape
    
    gridY = torch.linspace(-1, 1, steps = h).view(1, -1, 1, 1).expand(bsz, h, w, 1).to(imgs.device)
    gridX = torch.linspace(-1, 1, steps = w).view(1, 1, -1, 1).expand(bsz, h, w, 1).to(imgs.device)
    flow_grid = torch.cat((gridX, gridY), dim=3).type(torch.float32).to(imgs.device)
    # [TODO]: process the grid to rotate the image 
    
    # Convert angles to radians
    theta = new_angles * (torch.pi / 180)
    
    # Create rotation matrices for the entire batch
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    rotation_matrix = torch.stack([
        torch.stack([cos_theta, -sin_theta], dim=1),
        torch.stack([sin_theta, cos_theta], dim=1)
    ], dim=2).to(imgs.device)  # Shape: (bsz, 2, 2)
    
    # Reshape the grid to (bsz, h*w, 2) for matrix multiplication
    flow_grid = flow_grid.view(bsz, h * w, 2)
    
    # Apply rotation matrix to the grid
    rotated_grid = torch.bmm(flow_grid, rotation_matrix)  # Shape: (bsz, h*w, 2)
    
    # Reshape back to (bsz, h, w, 2)
    rotated_grid = rotated_grid.view(bsz, h, w, 2)
    # Apply the grid to the image
    rotated_imgs = F.grid_sample(imgs.unsqueeze(1), rotated_grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    rotated_imgs = rotated_imgs.squeeze(1)  # Remove the channel dimension
    rotated_imgs = torch.clamp(rotated_imgs, 0, 1)  # Ensure pixel values are between 0 and 1
    
    # Move the center back to the original position
    rotated_imgs = mv_center_by_pos(rotated_imgs, ori_center_x, ori_center_y)
    
    return rotated_imgs


def harden_img(imgs, threshold=0.1, scale=1.3):
    # imgs: bsz, (1), h, w
    imgs = torch.where(imgs < threshold, torch.tensor(0.0).to(imgs.device), imgs)
    imgs = imgs * scale
    imgs = torch.clamp(imgs, 0, 1)
    return imgs

def thicken_img(imgs):
    # this is ok
    imgs = imgs.squeeze(1)
    kernel = torch.tensor([[0.1, 0.3, 0.1],
                         [0.3, 1, 0.3],
                         [0.1, 0.3, 0.1]]).float().to(imgs.device)
    # kernel = kernel / kernel.sum()
    thicken_imgs = nn.functional.conv2d(imgs.unsqueeze(1), kernel.unsqueeze(0).unsqueeze(0), padding=1)
    thicken_imgs = thicken_imgs.squeeze(1)
    thicken_imgs = torch.clamp(thicken_imgs, 0, 1)  # Ensure pixel values are between 0 and 1
    return thicken_imgs

def thinnen_img(imgs):
    imgs = imgs.squeeze(1)
    threshold1 = 0.1
    threshold2 = 1 * 9
    new_img = torch.zeros_like(imgs)
    # for i in range(img.shape[0]):
    #     for j in range(img.shape[1]):
    #         neighbor = img[i-1:i+2, j-1:j+2]
    #         neighbors_sum = neighbor.sum()
    #         new_img[i][j] = img[i][j] * min(1, neighbors_sum / threshold2)
    kernel = torch.tensor([[1, 1, 1],
                          [1, 1, 1],
                          [1, 1, 1]], dtype=torch.float32).to(imgs.device)
    neighborhood = nn.functional.conv2d(imgs.unsqueeze(1), kernel.unsqueeze(0).unsqueeze(0), padding=1).squeeze(1)
    neighborhood = neighborhood / threshold2
    neighborhood = torch.clip(neighborhood, 0, 1)  # Ensure pixel values are between 0 and 1

    new_img = imgs * neighborhood
    new_img = torch.where(new_img < threshold1, torch.tensor(0.0).to(imgs.device), new_img)
    new_img = torch.clamp(new_img, 0, 1)  # Ensure pixel values are between 0 and 1
    new_img = harden_img(new_img)
    return new_img




def augment_dataset(dataset, num_data=30000, with_rot = False):
    bsz = 64
    data_loader = DataLoader(dataset, batch_size=bsz, pin_memory=True,
                            drop_last=False, shuffle=True, num_workers=8)
    augmented_data = []
    for i, (imgs, labels) in enumerate(data_loader):
        if len(augmented_data*bsz) >= num_data*10:
            break
        if i % 20 == 0:
            print(f"Processing batch {i}...")
        imgs = imgs.to(device)
        labels = labels.to(device)
        rot_range = 4 if with_rot else 1
        for rot_type in range(rot_range):
            expand_rot_type = torch.tensor([rot_type], dtype=torch.long).expand(imgs.shape[0]).to(device)
            rot_imgs = rotate_img(imgs, expand_rot_type)
            for scale_type in range(3):
                expand_scale_type = torch.tensor([scale_type], dtype=torch.long).expand(imgs.shape[0]).to(device)
                rescale_imgs = scale_img(rot_imgs, expand_scale_type)
                for pos_type in range(9):
                    if(random.random() > 0.7):
                        continue
                    expand_pos_type = torch.tensor([pos_type], dtype=torch.long).expand(imgs.shape[0]).to(device)
                    # print(imgs.shape, labels.shape, expand_pos_type.shape)
                    mv_imgs = mv_center(rescale_imgs, expand_pos_type)

                    thinned_imgs = mv_imgs.clone()
                    for thin_times in range(2):
                        thickness_types = get_thickness_label(thinned_imgs)
                        if with_rot:
                            new_lbls = torch.stack([labels, expand_rot_type, expand_pos_type, expand_scale_type, thickness_types], dim=1)
                        else:
                            new_lbls = torch.stack([labels, expand_pos_type, expand_scale_type, thickness_types], dim=1)
                        augmented_data.append((thinned_imgs, new_lbls))
                        thinned_imgs = thinnen_img(thinned_imgs)
                    thicked_imgs = mv_imgs.clone()
                    for thick_times in range(1):
                        thicked_imgs = thinnen_img(thicked_imgs)
                        thickness_types = get_thickness_label(thicked_imgs)
                        if with_rot:
                            new_lbls = torch.stack([labels, expand_rot_type, expand_pos_type, expand_scale_type, thickness_types], dim=1)
                        else:
                            new_lbls = torch.stack([labels, expand_pos_type, expand_scale_type, thickness_types], dim=1)
                        augmented_data.append((thicked_imgs, new_lbls))
        
    # augmented data (batchs, bsz, )

    augmented_imgs = torch.stack([img for img, lbl in augmented_data], dim=0)
    augmented_labels = torch.stack([lbl for img, lbl in augmented_data], dim=0)
    # print(augmented_imgs.shape, augmented_labels.shape)

    num_batches = augmented_imgs.shape[0]
    bsz = augmented_imgs.shape[1]
    augmented_imgs = augmented_imgs.view(num_batches * bsz, 1, 28, 28)
    augmented_labels = augmented_labels.view(num_batches * bsz, 4+with_rot)
    # print(augmented_imgs.shape, augmented_labels.shape)

    # remove those (scale_type=0 and thickness_types=2) or (thickness_types=4)
    removed_idx = (augmented_labels[:, 2] == 0 )*( augmented_labels[:, 3] == 2)
    removed_idx = removed_idx | (augmented_labels[:, 3] == 4)

    augmented_imgs = augmented_imgs[~removed_idx]
    augmented_labels = augmented_labels[~removed_idx]
    print(augmented_imgs.shape, augmented_labels.shape)

    # pick up num_data data
    if augmented_imgs.shape[0] > num_data:
        idx = random.sample(range(augmented_imgs.shape[0]), num_data)
        augmented_imgs = augmented_imgs[idx]
        augmented_labels = augmented_labels[idx]
    print(augmented_imgs.shape, augmented_labels.shape)
    # save to file
    if with_rot:
        torch.save(augmented_imgs, "augmented_imgs_rot.pt")
        torch.save(augmented_labels, "augmented_labels_rot.pt")
    else:
        torch.save(augmented_imgs, "augmented_imgs.pt")
        torch.save(augmented_labels, "augmented_labels.pt")

    new_dataset = {'imgs': augmented_imgs, 'labels': augmented_labels}
    return new_dataset

augment_dataset(train_set, num_data=100000, with_rot=True)


# train_loader = DataLoader(train_set, batch_size=8)

# # # get first batch
# for i, (imgs, labels) in enumerate(train_loader):
#     if i == 0:
#         break
# imgs = imgs.to(device)
# # ori_centers = get_center(imgs)
# mv_imgs = mv_center(imgs, [0, 1, 2, 3, 4, 5, 6, 7])

# rotate_imgs = rotate_img(mv_imgs, [0, 1, 2, 3, 0, 1, 2, 3])

# for i in range(8):
#     plt.subplot(3, 8, i + 1)
#     plt.imshow(imgs[i].squeeze(0).cpu().numpy(), cmap='gray')
#     plt.axis('off')
#     plt.subplot(3, 8, i + 9)
#     plt.imshow(mv_imgs[i].squeeze(0).cpu().numpy(), cmap='gray')
#     plt.axis('off')
#     plt.subplot(3, 8, i + 17)   
#     plt.imshow(rotate_imgs[i].squeeze(0).cpu().numpy(), cmap='gray')
#     plt.axis('off')
# plt.show()

# # # rescale_imgs =imgs
# move_center_imgs = mv_center(imgs, [0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6])

# # new_centers = get_center(move_center_imgs)
# # # print(f"new_centers={new_centers}")


# # thickness = get_thickness(imgs)
# # print(f"thickness={thickness}")

# # thicken_imgs = thicken_img(move_center_imgs)
# # thickness = get_thickness(thicken_imgs)
# # print(f"thickness={thickness}")

# # thinnen_imgs = thinnen_img(imgs)
# # thickness = get_thickness_label(imgs)
# # print(f"thickness={thickness}")

# for i in range(16):
#     plt.subplot(3, 16, i + 1)
#     plt.imshow(imgs[i].squeeze(0).cpu().numpy(), cmap='gray')
#     plt.axis('off')
#     plt.subplot(3, 16, i + 17)
#     plt.imshow(rescale_imgs[i].squeeze(0).cpu().numpy(), cmap='gray', alpha=0.5)
#     plt.axis('off')
#     plt.subplot(3, 16, i + 33)
#     plt.imshow(move_center_imgs[i].squeeze(0).cpu().numpy(), cmap='gray')
#     plt.axis('off')
    

# plt.show()
    