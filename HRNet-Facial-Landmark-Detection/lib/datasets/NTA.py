import os
import random

import torch
import torch.utils.data as data
import pandas as pd
from PIL import Image, ImageFile
import numpy as np
import math

from ..utils.transforms import fliplr_joints, crop, generate_target, transform_pixel


class NTA(data.Dataset):

    def __init__(self, cfg, is_train=True, transform=None):
        if is_train:
            self.csv_file = cfg.DATASET.TRAINSET
        else:
            self.csv_file = cfg.DATASET.TESTSET

        self.is_train = is_train
        self.transform = transform
        self.data_root = cfg.DATASET.ROOT
        self.input_size = cfg.MODEL.IMAGE_SIZE
        self.output_size = cfg.MODEL.HEATMAP_SIZE
        self.sigma = cfg.MODEL.SIGMA
        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rot_factor = cfg.DATASET.ROT_FACTOR
        self.label_type = cfg.MODEL.TARGET_TYPE
        self.flip = cfg.DATASET.FLIP

        # load annotations
        self.landmarks_frame = pd.read_csv(self.csv_file)

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.append_size = 1 / 4
        self.NUM_PTS = 194
        self.CROP_SIZE = 256

        #  увеличение размера рамки лица (для дополнительной информации)
        self.landmarks_frame['width_bbox'] = abs(self.landmarks_frame['bottom_x'] - self.landmarks_frame['top_x'])
        self.landmarks_frame['height_bbox'] = abs(self.landmarks_frame['bottom_y'] - self.landmarks_frame['top_y'])
        self.landmarks_frame['top_x'] -= self.append_size * self.landmarks_frame['width_bbox']
        self.landmarks_frame['top_y'] -= self.append_size * self.landmarks_frame['height_bbox']
        self.landmarks_frame['bottom_x'] += self.append_size * self.landmarks_frame['width_bbox']
        self.landmarks_frame['bottom_y'] += self.append_size * self.landmarks_frame['height_bbox']
        self.landmarks_frame['width_bbox'] = abs(self.landmarks_frame['bottom_x'] - self.landmarks_frame['top_x'])
        self.landmarks_frame['height_bbox'] = abs(self.landmarks_frame['bottom_y'] - self.landmarks_frame['top_y'])

        if self.is_train:
            # координаты меток относительно вернего угла бокса прямоугольника
            for id_point in range(self.NUM_PTS):
                self.landmarks_frame[f'Point_M{id_point}_X'] -= self.landmarks_frame['top_x']
                self.landmarks_frame[f'Point_M{id_point}_Y'] -= self.landmarks_frame['top_y']

            self.df_landmarks = self.landmarks_frame.drop(
                ['filename', 'top_x', 'top_y', 'bottom_x', 'bottom_y', 'width_bbox', 'height_bbox'], axis=1)

    def __len__(self):
        return len(self.landmarks_frame['filename'])

    def __getitem__(self, idx):

        row_bbox = self.landmarks_frame.loc[idx]
        row_landmarks = np.zeros(self.NUM_PTS * 2)
        if self.is_train:
            row_landmarks = np.array(self.df_landmarks.loc[idx].tolist())

        img_file = os.path.join(self.data_root, row_bbox['filename'])
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
        pts = row_landmarks.astype(np.int16).reshape(-1, 2)


        img = np.array(img.convert('RGB'), dtype=np.float32)
        # Image.fromarray(np.uint8(img)).show()

        xmin = np.min(pts[:, 0])
        xmax = np.max(pts[:, 0])
        ymin = np.min(pts[:, 1])
        ymax = np.max(pts[:, 1])

        center_w = (math.floor(xmin) + math.ceil(xmax)) / 2.0
        center_h = (math.floor(ymin) + math.ceil(ymax)) / 2.0

        center_w = 256/2.0
        center_h = 256/2.0

        scale = max(math.ceil(xmax) - math.floor(xmin), math.ceil(ymax) - math.floor(ymin)) / 200.0
        scale = 1
        center = torch.Tensor([center_w, center_h])

        scale *= 1.25

        nparts = pts.shape[0]
        # print(nparts)
        target = np.zeros((nparts, self.output_size[0], self.output_size[1]))
        tpts = pts.copy()
        tpts = tpts/4   # 256/4
        # print(tpts)
        for i in range(nparts):
            if tpts[i, 1] > 0:
                target[i], is_generate = generate_target(target[i], tpts[i] - 1, self.sigma,
                                                label_type=self.label_type)
                # if not is_generate:
                #     print(pts[i,1], img_file)
                #     print(row_landmarks[row_landmarks > 256])
                #     Image.fromarray(np.uint8(img)).show()

        # print(target.shape)
        # tr = target.sum(axis=0)
        # tr = np.concatenate(target[:5], axis=1)
        # Image.fromarray(np.uint8(tr*255)).show()
        # im = Image.fromarray(np.uint8(target[0]))
        # im.show()

        img = img.astype(np.float32)
        img = (img / 255.0 - self.mean) / self.std
        img = img.transpose([2, 0, 1])
        target = torch.Tensor(target)
        tpts = torch.Tensor(tpts)
        # center = torch.Tensor(center)

        meta = {'index': idx,
                'center': center,
                'scale': scale,
                'pts': torch.Tensor(pts),
                'tpts': tpts,
                "file_name": row_bbox['filename'],
                "crop_margin_x": margin_w,
                "crop_margin_y": margin_h,
                "scale_coef": f,
                "top_x": row_bbox['top_x'],
                "top_y": row_bbox['top_y']
                }

        return img, target, meta



