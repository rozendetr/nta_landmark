from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from scipy import interpolate
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import cv2,os
from scipy import ndimage
from scipy.ndimage.morphology import grey_dilation
import math
from PIL import Image
from utils.image import draw_umich_gaussian, draw_boundary

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class TransformByKeys(object):
    def __init__(self, transform, names):
        self.transform = transform
        self.names = set(names)

    def __call__(self, sample):
        for name in self.names:
            if name in sample:
                sample[name] = self.transform(sample[name])

        return sample


class NTA_Dataset(Dataset):
    def __init__(self, root_img, dataframe, transforms=None, CROP_SIZE=256, NUM_PTS=194):
        super(NTA_Dataset, self).__init__()
        self.root = root_img
        self.df = dataframe.copy()
        self.transforms = transforms
        self.CROP_SIZE = CROP_SIZE
        self.NUM_PTS = NUM_PTS
        self.append_size = 1 / 4
        self.hmap_size = 64

        #  увеличение размера рамки лица (для дополнительной информации)
        self.df['width_bbox'] = abs(self.df['bottom_x'] - self.df['top_x'])
        self.df['height_bbox'] = abs(self.df['bottom_y'] - self.df['top_y'])
        self.df['top_x'] -= self.append_size * self.df['width_bbox']
        self.df['top_y'] -= self.append_size * self.df['height_bbox']
        self.df['bottom_x'] += self.append_size * self.df['width_bbox']
        self.df['bottom_y'] += self.append_size * self.df['height_bbox']
        self.df['width_bbox'] = abs(self.df['bottom_x'] - self.df['top_x'])
        self.df['height_bbox'] = abs(self.df['bottom_y'] - self.df['top_y'])

        # координаты меток относительно вернего угла бокса прямоугольника
        for id_point in range(self.NUM_PTS):
            self.df[f'Point_M{id_point}_X'] -= self.df['top_x']
            self.df[f'Point_M{id_point}_Y'] -= self.df['top_y']

        self.df_landmarks = self.df.drop(
            ['filename', 'top_x', 'top_y', 'bottom_x', 'bottom_y', 'width_bbox', 'height_bbox'], axis=1)

    def __len__(self):
        return len(self.df['filename'])

    def __getitem__(self, idx):
        sample = {}
        row_bbox = self.df.loc[idx]
        row_landmarks = np.array(self.df_landmarks.loc[idx].tolist())

        img_file = os.path.join(self.root, row_bbox['filename'])
        img = Image.open(img_file)

        # кроп лица
        bbox = [row_bbox['top_x'], row_bbox['top_y'], row_bbox['bottom_x'], row_bbox['bottom_y']]
        img = img.crop(bbox)

        # ресайз кропа до размеров CROP_SIZE с сохранением соотношения сторон
        w, h = img.size
        if h > w:
            f = self.CROP_SIZE / w
        else:
            f = self.CROP_SIZE / h
        img = img.resize((int(w * f), int(h * f)))
        row_landmarks = row_landmarks * f

        # CropCenter
        w, h = img.size
        margin_h = (h - self.CROP_SIZE) // 2
        margin_w = (w - self.CROP_SIZE) // 2
        img = img.crop([margin_w, margin_h, self.CROP_SIZE + margin_w, self.CROP_SIZE + margin_h])
        row_landmarks = row_landmarks.astype(np.int16).reshape(-1, 2)
        row_landmarks -= np.array((margin_w, margin_h), dtype=np.int16)[None, :]
        # row_landmarks = row_landmarks.reshape(-1)

        # hmap = np.zeros((self.NUM_PTS + 1, self.hmap_size, self.hmap_size), dtype=np.float32)
        # M = np.zeros((self.NUM_PTS + 1, self.hmap_size, self.hmap_size), dtype=np.float32)

        hmap = np.zeros((self.NUM_PTS, self.hmap_size, self.hmap_size), dtype=np.float32)
        M = np.zeros((self.NUM_PTS, self.hmap_size, self.hmap_size), dtype=np.float32)

        for ind, xy in enumerate(row_landmarks):
            hmap[ind] = draw_umich_gaussian(hmap[ind], xy / self.CROP_SIZE * self.hmap_size, 7)
        # hmap[-1] = draw_boundary(hmap[-1], np.clip((row_landmarks / self.CROP_SIZE * self.hmap_size).astype(np.int), 0, self.NUM_PTS))

        for i in range(len(M)):
            M[i] = grey_dilation(hmap[i], size=(3, 3))
        M = np.where(M >= 0.5, 1, 0)


        sample = {"file_name": row_bbox['filename'],
                  "image": img,
                  "landmarks": torch.from_numpy(row_landmarks.astype(np.float32)),
                  "crop_margin_x": margin_w,
                  "crop_margin_y": margin_h,
                  "scale_coef": f,
                  "top_x": row_bbox['top_x'],
                  "top_y": row_bbox['top_y'],
                  "hmap": hmap,
                  "M": M}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample




if __name__=="__main__":
    import sys, random
    from utils.image import draw_umich_gaussian, draw_boundary


    DIR_TRAIN = '../../data/train'
    DIR_TRAIN_IMAGES = "../../data/train/images"

    df_landmarks = pd.read_csv(os.path.join(DIR_TRAIN, 'landmarks_train.csv'))

    NUM_PTS = 194
    CROP_SIZE = 256

    # train_transforms = transforms.Compose([
    #     TransformByKeys(transforms.Grayscale(num_output_channels=1), ("image",)),
    #     TransformByKeys(transforms.ToTensor(), ("image",)),
    #     TransformByKeys(transforms.Normalize(mean=[0.5], std=[0.225]), ("image",)),
    # ])


    dataset = NTA_Dataset(DIR_TRAIN_IMAGES, df_landmarks)
    l = len(dataset)
    sample = dataset[random.randint(0,l-1)]
    img = sample['image']
    hmap = sample['hmap']
    M = sample['M']
    landmarks = sample['landmarks']

    plt.subplot(3, 1, 1)
    plt.imshow(img)
    plt.imshow(cv2.resize(hmap[-1], dsize=(256, 256), interpolation=cv2.INTER_AREA),alpha=0.3)
    plt.subplot(3, 1, 2)
    plt.imshow(np.max(hmap, axis=0))
    plt.subplot(3, 1, 3)
    plt.imshow(np.max(M, axis=0))


    plt.show()

    print(landmarks)
