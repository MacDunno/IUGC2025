# --------------------------------------------------------
# Copyright (c) 2022 Megvii, Inc. and its affiliates. All Rights Reserved.
# ---------------------------------------------------------
""" 
US1 Datasets. 
"""
import os.path as osp
from torchvision.datasets import ImageFolder
from ..registry import DATASETS
from mae_lite.utils import get_root_dir

@DATASETS.register()
class US1(ImageFolder):
    def __init__(self, train, transform=None):
        root = osp.join(
            get_root_dir(),
            "data/US1/US1_{}".format("train" if train else "val")
        )
        super(US1, self).__init__(root, transform=transform)
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target
