import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from datasets.NTA import *
from models.hourglass import hg
from losses.loss import Loss_weighted
import cv2
import os
import numpy as np

#basic setting
NUM_PTS = 194
CROP_SIZE = 256
DIR_TRAIN = '../data/train'
DIR_TRAIN_IMAGES = "../data/train/images"

df_landmarks = pd.read_csv(os.path.join(DIR_TRAIN, 'landmarks_train.csv'))

num_epochs = 240
vis_result = True
batch_size = 4
start_lr = 1e-6
if(vis_result):
    import matplotlib.pyplot as plt
W = 5
omega = 300
epsilon = 2

pretrained = False


#model load
# if pretrained:
#     model = torch.load(pretrain_path)
# else:
model  = hg(num_stacks=2,
            num_blocks=1,
            num_classes=NUM_PTS)
model = model.cuda()
print("model loaded")

train_transforms = transforms.Compose([
        # TransformByKeys(transforms.Grayscale(num_output_channels=1), ("image",)),
        TransformByKeys(transforms.ToTensor(), ("image",)),
        TransformByKeys(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ("image",)),
    ])

#dataset load
train_dataset = NTA_Dataset(DIR_TRAIN_IMAGES, df_landmarks, transforms = train_transforms)
dataiter = len(train_dataset)//batch_size
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           shuffle=True, num_workers=0)
print("images : ",len(train_dataset))

#loss define
criterion = Loss_weighted()

#optim setting
optimizer = optim.RMSprop(model.parameters(), lr=start_lr, weight_decay=1e-5, momentum=0)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer,  milestones=[60,120], gamma=0.1)

#train start!
print("train start!")

for epoch in range(num_epochs):
    losses = list()
    for iteration, sample in enumerate(train_loader):
        img = sample['image']
        hmap = sample['hmap']
        M = sample['M']
        pts = sample['landmarks']
        # img = img.permute(0, 3, 1, 2)

        print(img.size())
        img, hmap, M = img.cuda(), hmap.cuda(), M.cuda()

        out = model(img)

        loss = sum(criterion(o, hmap, M) for o in out)
        losses.append(loss.item())
        print(str(epoch)," :: ",str(iteration), "/",dataiter,"\n  loss     :: ",loss.item())
        print("  avg loss :: ",sum(losses)/len(losses))

        # Backward pass and parameter update.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if vis_result and (iteration % 20 == 0):
            y_hat = out[-1][0,:].detach().cpu().numpy()
            plt.imshow(img.permute(0,2,3,1)[0].cpu().numpy())
            plt.imshow(cv2.resize(np.max(y_hat, axis=0), dsize=(256, 256), interpolation=cv2.INTER_LINEAR),alpha=0.5)
            plt.show(block=False)
            plt.pause(3)
            plt.close()

    if(epoch %3 == 0):
        torch.save(model, "./ckpt/"+str(epoch)+".pth")

    scheduler.step()
