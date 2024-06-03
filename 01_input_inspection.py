#! /usr/bin/env python
# the script provided on coursera does not work
# becase some training images are not the same size as the masks
# it looks like it is jut the matter of scaling, not sure how that happened

import cv2

import pandas as pd

# import helper
import settings
from utils.utils import quick_plot


def printall(idx, image_path, mask_path, image, mask, i_ratio, m_ratio):
    print(idx)
    print(image_path, mask_path)
    print(image.shape[:2], mask.shape)
    print(f"{i_ratio}   {m_ratio}")


def main():
    print(settings.FILE_PATHS)
    df = pd.read_csv(settings.FILE_PATHS)
    total = 0
    size_mismatch = 0
    ratio_mismatch = 0
    for idx, row in df.iterrows():
        image_path = row['images']
        mask_path  = row['masks']
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255.0

        total += 1
        if image.shape[:2] != mask.shape:
            # image is ndarray now - the number of columns is the first dime
            [i_height, i_width] = image.shape[:2]
            [m_height, m_width] = mask.shape[:2]
            size_mismatch += 1
            i_ratio = round(i_width/i_height, 2)
            m_ratio = round(m_width/m_height, 2)
            if i_ratio != m_ratio:
                ratio_mismatch += 1
                # printall(idx, image_path, mask_path, image, mask, i_ratio, m_ratio)
                continue
            if i_width < m_width:
                # printall(idx, image_path, mask_path, image, mask, i_ratio, m_ratio)
                continue
            image = cv2.resize(image, (m_width, m_height) )
            printall(idx, image_path, mask_path, image, mask, i_ratio, m_ratio)
            quick_plot(image, mask)
    print()
    print()
    print(total, size_mismatch, ratio_mismatch)


################################
if __name__ == "__main__":
    main()
