from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from albumentations import Compose
from torch.utils.data import Dataset

from utils.utils import resize_if_needed


class SegmentationDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, augmenting_tfm: Compose):
        # aug tfm is a composition if several atomic transformations
        self.df = dataframe
        self.aug_tfm = augmenting_tfm

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = Path(row.images)
        mask_path =  Path(row.masks)

        for path in [image_path, mask_path]:
            if not path.exists():
                raise Exception(f"{path} not found")

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)  # (h,w,c)
        mask = np.expand_dims(mask, axis=-1)

        # in this particular test set, the original image is bigger that the mask
        # in almost 50% of the cases
        image = resize_if_needed(image, mask)
        if image is None:
            raise Exception(f"{image_path} resizing problem")

        if self.aug_tfm:
            data = self.aug_tfm(image=image, mask=mask)
            image = data["image"]
            mask = data["mask"]

        # For historical reasons, OpenCV reads an image in BGR format
        # (h,w,c) -> (c,h,w)
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        mask = np.transpose(mask, (2, 0, 1)).astype(np.float32)

        image = torch.Tensor(image) / 255.0
        mask = torch.round(torch.Tensor(mask) / 255.0)

        return image, mask
