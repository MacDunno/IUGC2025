import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image
import pandas as pd
import ast
import numpy as np
import random
import math


class HeatmapLandmarkDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, train=True, heatmap_size=64, sigma=4.0):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.train = train
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def generate_heatmap(self, center_x, center_y, height, width):
        x = np.arange(0, width, 1, np.float32)
        y = np.arange(0, height, 1, np.float32)
        y = y[:, np.newaxis]
        x0 = center_x
        y0 = center_y
        heatmap = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))
        return heatmap

    def _apply_geom_transform(self, img, keypoints, img_size=(512, 512)):
        if not self.train:
            return img, keypoints  # no change
        angle = random.uniform(-15.0, 15.0)   # degrees
        scale = random.uniform(0.9, 1.1)
        shear = [0.0, 0.0]
        translate = [0.0, 0.0]
        w, h = img_size  # expected 512, 512 after resize
        img = F.resize(img, [h, w])
        center = (w * 0.5, h * 0.5)
        img = F.affine(img, angle=angle, translate=translate, scale=scale, shear=shear, interpolation=F.InterpolationMode.BILINEAR, center=center)
        theta = math.radians(angle)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        cx, cy = center
        def transform_point(x, y):
            x0 = x - cx
            y0 = y - cy
            xr = scale * (x0 * cos_t - y0 * sin_t)
            yr = scale * (x0 * sin_t + y0 * cos_t)
            xt = xr + cx
            yt = yr + cy
            xt = max(0.0, min(xt, w - 1))
            yt = max(0.0, min(yt, h - 1))
            return [xt, yt]
        kp_ps1 = transform_point(keypoints[0][0], keypoints[0][1])
        kp_ps2 = transform_point(keypoints[1][0], keypoints[1][1])
        kp_aop = transform_point(keypoints[2][0], keypoints[2][1])
        return img, [kp_ps1, kp_ps2, kp_aop]

    def _apply_color_transform(self, img):
        if not self.train:
            return img
        gamma = random.uniform(0.8, 1.2)
        img = F.adjust_gamma(img, gamma)
        contrast = random.uniform(0.8, 1.2)
        img = F.adjust_contrast(img, contrast)
        return img

    def __getitem__(self, index):
        row = self.data.iloc[index]
        filename = row['Filename']
        img_path = os.path.join(self.img_dir, filename)
        image = Image.open(img_path).convert('RGB')
        image = transforms.functional.resize(image, (512, 512))
        ps1 = ast.literal_eval(row["PS1"])
        ps2 = ast.literal_eval(row["PS2"])
        aop = ast.literal_eval(row["FH1"])
        keypoints = [ps1, ps2, aop]
        image, keypoints = self._apply_geom_transform(image, keypoints, img_size=(512, 512))
        image = self._apply_color_transform(image)
        image = self.transform(image)
        img_width, img_height = 512, 512
        heatmaps = np.zeros((3, self.heatmap_size, self.heatmap_size), dtype=np.float32)
        scale_x = self.heatmap_size / img_width
        scale_y = self.heatmap_size / img_height
        for i, kp in enumerate(keypoints):
            x = int(kp[0] * scale_x)
            y = int(kp[1] * scale_y)
            x = max(0, min(x, self.heatmap_size - 1))
            y = max(0, min(y, self.heatmap_size - 1))
            heatmaps[i] = self.generate_heatmap(x, y, self.heatmap_size, self.heatmap_size)
        heatmaps = torch.from_numpy(heatmaps)
        landmarks = [
            keypoints[0][0] / img_width, keypoints[0][1] / img_height,
            keypoints[1][0] / img_width, keypoints[1][1] / img_height,
            keypoints[2][0] / img_width, keypoints[2][1] / img_height
        ]
        landmarks = torch.tensor(landmarks, dtype=torch.float32)
        return image, heatmaps, landmarks