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
from torch.utils.data import DataLoader
from tqdm import tqdm

import settings
from classes.segmentation_dataset import SegmentationDataset
from classes.segmentation_model import SegmentationModel
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


def get_train_augs():
    return A.Compose(
        [
            A.Resize(settings.IMG_SIZE, settings.IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
        ]
    )


def get_valid_augs():
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


def train_fn(data_loader, model, optimizer):
    model.train()
    total_loss = 0.0

    # tqdm is a progress meter
    for images, masks in tqdm(data_loader):

        images = images.to(settings.DEVICE)
        masks = masks.to(settings.DEVICE)

        optimizer.zero_grad()
        logits, loss = model(images, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        return total_loss / len(data_loader)


def eval_fn(data_loader, model):

    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for images, masks in tqdm(data_loader):

            images = images.to(settings.DEVICE)
            masks = masks.to(settings.DEVICE)

            logits, loss = model(images, masks)

            total_loss += loss.item()

    return total_loss / len(data_loader)


def main():

    df = pd.read_csv(settings.FILE_PATHS)
    print(f"original df size:", df.size)
    df = sanitize(df)
    print(f"sanitized df size:", df.size)

    # split the input
    train_df, valid_df = train_test_split(df, test_size=0.2, random_state=88)

    # create the data set (load + augment)
    trainset = SegmentationDataset(train_df, get_train_augs())
    validset = SegmentationDataset(valid_df, get_valid_augs())

    # inspect(trainset)

    # load dataset into batches
    trainloader = DataLoader(trainset, batch_size=settings.BATCH_SIZE, shuffle=True)
    validloader = DataLoader(validset, batch_size=settings.BATCH_SIZE)

    model = SegmentationModel()
    model.to(settings.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=settings.LR)

    best_valid_loss = np.Inf
    for i in range(settings.EPOCHS):
        train_loss = train_fn(trainloader, model, optimizer)
        valid_loss = eval_fn(validloader, model)

        if valid_loss < best_valid_loss:
            torch.save(model.state_dict(), "bestModel.pt")
            print("SAVED")
            best_valid_loss = valid_loss
        print(f"Epoch : {i+1} train_loss :{train_loss} valid_loss :{valid_loss}")

    model.load_state_dict(torch.load("bestModel.pt"))
    for idx in range(len(validset)):
        image, mask = validset[idx]
        logits_mask = model(
            image.to(settings.DEVICE).unsqueeze(0)
        )  # (C, H, W) -> (1, C, H, W)
        pred_mask = torch.sigmoid(logits_mask)
        pred_mask = (pred_mask > 0.5) * 1.0

        quick_plot(image, pred_mask.detach().cpu().squeeze(0))


################################
if __name__ == "__main__":
    main()
