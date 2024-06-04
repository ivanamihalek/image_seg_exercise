from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt
from torch import Tensor

def paths_ok(paths) -> bool:
    for path in paths:
        if path.exists(): continue
        print(f"{path} not found")
        return False

    return True


def quick_plot(img: np.ndarray | Tensor, mask: np.ndarray | Tensor, predicted_mask:  np.ndarray | Tensor | None = None):

    no_of_subplots = 2 if predicted_mask is None else 3
    f, ax = plt.subplots(1, no_of_subplots, figsize=(10, 5))

    ax[0].set_title('IMAGE')
    if img.shape[0] == 3:
        img = img.permute(1, 2, 0).squeeze()
    ax[0].imshow(img)

    ax[1].set_title('GROUND TRUTH')
    if mask.shape[0] == 1:
        mask = mask.permute(1, 2, 0).squeeze()
    ax[1].imshow(mask, cmap='gray')

    if predicted_mask is not None:
        ax[2].set_title('PREDICTED')
        if predicted_mask.shape[0] == 1:
            predicted_mask = predicted_mask.permute(1, 2, 0).squeeze()
        ax[2].imshow(predicted_mask, cmap='gray')

    plt.show()


def resize_if_needed(image: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """ Return resized image if everything is sane, ele None"""
    if image.shape[:2] == mask.shape: return image
    # image is ndarray now - the number of columns is the first dime
    return cv2.resize(image, (mask.shape[1], mask.shape[0]))


def read_pair(image_path: Path, mask_path: Path) -> tuple[np.ndarray, np.ndarray]:
    image: np.ndarray = cv2.imread(str(image_path))
    mask: np.ndarray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)  # (h,w,c)
    return image, mask


def shape_compatible(image: np.ndarray, mask: np.ndarray) -> bool:
    [i_height, i_width] = image.shape[:2]
    [m_height, m_width] = mask.shape[:2]

    i_ratio = round(i_width / i_height, 2)
    m_ratio = round(m_width / m_height, 2)
    if i_ratio != m_ratio:
        print(f"asp ratio image: {i_ratio}   asp ratio mask {m_ratio}")
        return False  # we are not dealing with this
    if i_width < m_width:
        print(f"orig image smaller than the mask")
        return False  # we are not dealing with this
    return True
