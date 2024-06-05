#! /usr/bin/env python
# the script provided on coursera does not work
# becase some training images are not the same size as the masks
# it looks like it is jut the matter of scaling, not sure how that happened
from pathlib import Path
from pprint import pprint
from time import time

import albumentations as A
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
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


def model_run(encoder, weights, trainloader, validationloader, testset):
    print("\n********************************************************")
    print(f"running encoder {encoder}  with weights {weights}")
    time0 = time()
    model = SegmentationModel(encoder=encoder, weights=weights)
    model.to(settings.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=settings.LR)

    # training loop
    best_validation_loss = np.Inf
    for i in range(settings.EPOCHS):
        train_loss = train_fn(trainloader, model, optimizer)
        validation_loss = eval_fn(validationloader, model)
        saved = False
        if validation_loss < best_validation_loss:
            torch.save(model.state_dict(), "bestModel.pt")
            saved = True
            best_validation_loss = validation_loss
        msg = f"Epoch : {i+1} train_loss: {train_loss:.2f}   validation_loss: {validation_loss:.2f} "
        msg += "SAVED" if saved else "      "  # for cleanup after \r
        print(msg, end='\r', flush=True)
    print()
    # results
    testloader =  DataLoader(testset, batch_size=settings.BATCH_SIZE)
    test_loss =  eval_fn(testloader, model)
    print(f"\ntest  loss: {test_loss:.2f}      time: {(time()-time0)/60:.2f} min\n")


def find_possible_encoders() -> dict:
    """
     according to https://github.com/jlcsilva/segmentation_models.pytorch/blob/master/docs/encoders.rst?plain=1
     which is the source for this https://segmentation-models-pytorch.readthedocs.io/en/latest/encoders.html
     the following models are small enough (4M params) to fit into my GPU
     | timm-regnetx_002   | imagenet   | 2M          |
     +---------------------+------------+-------------+
     | timm-regnetx_004   | imagenet   | 4M          |
     +------------------------+--------------------------------------+-------------+
      timm-efficientnet-lite0| imagenet                             | 4M          |
     +------------------------+--------------------------------------+-------------+
     | timm-efficientnet-lite1| imagenet                             | 4M          |
     +------------------------+--------------------------------------+-------------+
     | efficientnet-b0        | imagenet                             | 4M          |
     | timm-efficientnet-b0   | imagenet / advprop / noisy-student   | 4M         |
     | mobilenet\_v2   | imagenet   | 2M          |
     +=====================+============+=============+
     """
    # for k, v in smp.encoders.encoders.items():
    #     print(k)
    #     # pprint(v)
    #     # v[]
    encoder_weights = {}
    for encoder_name in ['timm-regnetx_002', 'timm-regnetx_004',
                         'timm-efficientnet-lite0' , 'timm-efficientnet-lite1',
                        'efficientnet-b0', 'timm-efficientnet-b0', ' mobilenet_v2'
                         ]:
        if encoder_name not in smp.encoders.encoders: continue
        encoder_weights[encoder_name] = list(smp.encoders.encoders[encoder_name]['pretrained_settings'].keys())
    # (encoder_weights)
    return encoder_weights


def main():

    encoder_weights = find_possible_encoders()

    df = pd.read_csv(settings.FILE_PATHS)
    df = sanitize(df)

    # split the input
    train_df, validation_and_testing_df = train_test_split(df, test_size=0.4)  # , random_state=88)
    validation_df, testing_df =  train_test_split(validation_and_testing_df, test_size=0.5)  # , , random_state=88)

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
    for encoder, weights in encoder_weights.items():
        for weight in weights:
            model_run(encoder, weight, trainloader, validationloader, testset)

################################
if __name__ == "__main__":
    main()
