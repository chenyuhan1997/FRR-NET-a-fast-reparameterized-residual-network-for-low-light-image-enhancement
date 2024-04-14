import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.nn.functional as tnf
import torch.optim
import os
import argparse
import model
from loss import *
import numpy as np
from torchvision import transforms
import cv2
from model import *
import torch.optim as optim
import kornia
import scipy.io as scio
import matplotlib.pyplot as plt
from pytorch_msssim import ms_ssim

train_data_path = '.\\ours\\'

root_high = train_data_path + 'high\\'
root_low = train_data_path+'low\\'


train_path = '.\\Train_result\\'
device = "cuda"


batch_size = 10
epochs = 100
lr = 1e-3

Train_Image_Number=len(os.listdir(train_data_path+'high\\high'))
Iter_per_epoch=(Train_Image_Number % batch_size!=0)+Train_Image_Number//batch_size

transforms = transforms.Compose([
    transforms.CenterCrop(162),
    transforms.ToTensor(),
])




Data_high = torchvision.datasets.ImageFolder(root_high, transform=transforms)
dataloader_high = torch.utils.data.DataLoader(Data_high, batch_size, shuffle=False)

Data_low = torchvision.datasets.ImageFolder(root_low, transform=transforms)
dataloader_low = torch.utils.data.DataLoader(Data_low, batch_size, shuffle=False)



lowlight_enhancement = Fast_low_light_enhancement()
is_cuda = True
if is_cuda:
    lowlight_enhancement = lowlight_enhancement.cuda()
optimizer = optim.Adam(lowlight_enhancement.parameters(), lr = lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60, 80], gamma=0.1)
#-----------------------------------------------------------------------------------------------------------------------
loss_c = L_color()
L_L1 = nn.SmoothL1Loss(reduction='mean')
loss_s = L_spa()
loss_train=[]
loss_l1_train=[]
loss_color_train=[]
loss_spa_train=[]
loss_ssim_train=[]
lr_list=[]

for iteration in range(epochs):
    lowlight_enhancement.train()
    data_iter_high = iter(dataloader_high)
    data_iter_low = iter(dataloader_low)

    for step in range(Iter_per_epoch):
        data_high, _ = next(data_iter_high)
        data_low, _ = next(data_iter_low)

        if is_cuda:
            data_high = data_high.cuda()
            data_low = data_low.cuda()

        optimizer.zero_grad()
        end = lowlight_enhancement(data_low)

        loss_l1 = L_L1(end, data_high)
        loss_color = torch.mean(loss_c(end))
        loss_spa = torch.mean(loss_s(end, data_high))
        loss_ssim = 1-ms_ssim(end, data_high, data_range=1.0, size_average=True)
        loss = loss_l1 + loss_color + loss_spa + loss_ssim

        loss.backward()

        optimizer.step()
        loss_total = loss.item()

        print('Epoch/step: %d/%d, loss: %.7f, lr: %f' % (
        iteration + 1, step + 1, loss_total, optimizer.state_dict()['param_groups'][0]['lr']))
        loss_train.append(loss.item())
        loss_l1_train.append(loss_l1.item())
        loss_color_train.append(loss_color.item())
        loss_spa_train.append(loss_spa.item())
        loss_ssim_train.append(loss_ssim.item())
    scheduler.step()


    lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])

    # Save Weights and result
    torch.save({'weight': lowlight_enhancement.state_dict(), 'epoch': epochs},
               os.path.join(train_path, 'Encoder_weight.pkl'))

    scio.savemat(os.path.join(train_path, 'TrainData.mat'),
                 {'Loss': np.array(loss_train),
                  'loss_l1_train': np.array(loss_l1_train),
                  'loss_spa_train': np.array(loss_spa_train),
                  'loss_color_train': np.array(loss_color_train),
                  'loss_ssim_train': np.array(loss_ssim_train),
                  })
    scio.savemat(os.path.join(train_path, 'TrainData_plot_loss.mat'),
                 {'loss_train': np.array(loss_train),
                  'loss_l1_train': np.array(loss_l1_train),
                  'loss_color_train': np.array(loss_color_train),
                  'loss_spa_train': np.array(loss_spa_train),
                  'loss_ssim_train': np.array(loss_ssim_train),
                  })
# plot
    def Average_loss(loss):
        return [sum(loss[i * Iter_per_epoch:(i + 1) * Iter_per_epoch]) / Iter_per_epoch for i in
                range(int(len(loss) / Iter_per_epoch))]


    plt.figure(figsize=[12, 8])
    plt.subplot(2, 3, 1), plt.plot(Average_loss(loss_train)), plt.title('Loss')
    plt.subplot(2, 3, 2), plt.plot(Average_loss(loss_l1_train)), plt.title('loss_l1')
    plt.subplot(2, 3, 3), plt.plot(Average_loss(loss_color_train)), plt.title('loss_color')
    plt.subplot(2, 3, 4), plt.plot(Average_loss(loss_spa_train)), plt.title('loss_spa')
    plt.subplot(2, 3, 5), plt.plot(Average_loss(loss_ssim_train)), plt.title('loss_ssim')
    plt.tight_layout()
    plt.savefig(os.path.join(train_path, 'curve_per_epoch.jpg'))










