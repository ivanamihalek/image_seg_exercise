#! /usr/bin/env python
# the script provided on coursera does not work
# becase some training images are not the same size as the masks
# it looks like it is jut the matter of scaling, not sure how that happened
from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, ConcatDataset

import settings
from classes.segmentation_dataset import SegmentationDataset
from classes.segmentation_model import SegmentationModel
from utils.training import train_fn, eval_fn
from utils.utils import paths_ok, shape_compatible, read_pair, quick_plot
from random import sample


def sanitize(df: pd.DataFrame):
    bad_ids = []
    for idx, row in df.iterrows():
        image_path = Path(row["images"])
        mask_path = Path(row["masks"])
        if not paths_ok([image_path, mask_path]):
            bad_ids.append(idx)
        if not shape_compatible(*read_pair(image_path, mask_path)):
            print(f"shape compat problem: {image_path} {mask_path}")
            bad_ids.append(idx)
    return df.drop(bad_ids) if bad_ids else df


def complex_aug_tfm():
    return A.Compose(
        [
            A.Resize(settings.IMG_SIZE, settings.IMG_SIZE),
            A.HorizontalFlip(p=1.0),
            A.ShiftScaleRotate(p=0.5),
            A.VerticalFlip(p=0.5),
            A.OneOf(
                [
                    A.CLAHE(clip_limit=2),
                    A.RandomBrightnessContrast(),
                ],
                p=0.3,
            ),
        ]
    )

def complex_aug_tfm_2():
    return A.Compose(
        [
            A.Resize(settings.IMG_SIZE, settings.IMG_SIZE),
            A.ShiftScaleRotate(p=1.0),
            A.Blur(blur_limit=3),
            A.OpticalDistortion(),
            A.GridDistortion(),
            A.HueSaturationValue(),
         ]
    )


def simple_aug_tfm():
    return A.Compose(
        [
            A.Resize(settings.IMG_SIZE, settings.IMG_SIZE),
        ]
    )


def inspect(trainset):
    # note: df size is rws x colums, while len(df) is number of rows
    for idx in sample(range(len(trainset)), 20):
        print(idx, len(trainset))
        img, mask = trainset[idx]
        quick_plot(img, mask)


def main():

    df = pd.read_csv(settings.FILE_PATHS)
    print(f"original df size:", df.size)
    df = sanitize(df)
    print(f"sanitized df size:", df.size)

    # split the input
    train_df, validation_and_testing_df = train_test_split(df, test_size=0.4, random_state=88)
    validation_df, testing_df =  train_test_split(validation_and_testing_df, test_size=0.5, random_state=88)

    # create the data set (load + augment)
    trainset = ConcatDataset(
        [
            SegmentationDataset(train_df, simple_aug_tfm()),
            SegmentationDataset(train_df, complex_aug_tfm()),
            # SegmentationDataset(train_df, complex_aug_tfm_2()) # this does not improve the results
        ]
    )
    validationset = SegmentationDataset(validation_df, simple_aug_tfm())
    testset =  SegmentationDataset(testing_df, simple_aug_tfm())

    # inspect(trainset)

    # load dataset into batches
    trainloader = DataLoader(trainset, batch_size=settings.BATCH_SIZE, shuffle=True)
    validationloader = DataLoader(validationset, batch_size=settings.BATCH_SIZE)

    model = SegmentationModel(encoder=settings.ENCODER, weights=settings.WEIGHTS)
    model.to(settings.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=settings.LR)

    # training loop
    best_validation_loss = np.Inf
    for i in range(settings.EPOCHS):
        train_loss = train_fn(trainloader, model, optimizer, verbose=True)
        validation_loss = eval_fn(validationloader, model, verbose=True)

        if validation_loss < best_validation_loss:
            torch.save(model.state_dict(), "bestModel.pt")
            print("SAVED")
            best_validation_loss = validation_loss
        print(f"Epoch : {i+1} train_loss: {train_loss:.2f}   validation_loss: {validation_loss:.2f}")

    # results
    testloader =  DataLoader(testset, batch_size=settings.BATCH_SIZE)
    test_loss =  eval_fn(testloader, model)
    print(f"\n\ntest  loss: {test_loss:.2f}\n")
    model.load_state_dict(torch.load("bestModel.pt"))
    for idx in range(len(testset)):
        image, mask = testset[idx]
        logits_mask = model(
            image.to(settings.DEVICE).unsqueeze(0)
        )  # (C, H, W) -> (1, C, H, W)
        pred_mask = torch.sigmoid(logits_mask)
        pred_mask = (pred_mask > 0.5) * 1.0

        quick_plot(image, mask, pred_mask.detach().cpu().squeeze(0))


################################
if __name__ == "__main__":
    main()
