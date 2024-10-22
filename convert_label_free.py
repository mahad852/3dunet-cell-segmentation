import PIL.Image
import torchvision.transforms.transforms
from datasets.CellDatasetMIP import CellDatasetMIP
from datasets.CellDataset import CellDataset
from datasets.CellDataset2D import CellDataset2D
from datasets.AllenCellDataset import AllenCellDataset

import numpy as np
import matplotlib.pyplot as plt
import PIL
from skimage.filters import (threshold_otsu, threshold_niblack,
                             threshold_sauvola)
import skimage.io 
import tifffile

from utils.data_utils import read_czi_image, save_image_as_tiff, merge_channels
from utils.img_transformations import AddChannel, ResizeImage, CropImageDepth

from datasets.CZIDataset import CZIDataset

from torchvision.transforms import transforms

import os

def scale_image(img: np.ndarray, max_val = 65535) -> np.ndarray:
    return ((img - img.min())/(img.max() - img.min())) * max_val
    
transform_img = transforms.Compose([
    ResizeImage(size=(512, 512)),
    CropImageDepth(depth=26)#, center=23),
])


root_dir = "/home/mali2/datasets/CellSeg/"

train_ds = AllenCellDataset(os.path.join(root_dir, "AllenCellData"), targets=["microtubule", "mitochondria"], transform_image=transform_img, transform_seg=transform_img)
val_ds = AllenCellDataset(os.path.join(root_dir, "AllenCellData"), targets=["microtubule", "mitochondria"], transform_image=transform_img, transform_seg=transform_img, is_train=False)

for i, (imputs, labels) in enumerate(train_ds):
    labels = merge_channels(labels)
    print(f"Train; Merged: {i}")
    save_image_as_tiff(labels, os.path.join(root_dir, "LabelFreeCust", "train", f"pos{i}.tiff"))

for i, (imputs, labels) in enumerate(val_ds):
    labels = merge_channels(labels)
    print(f"Val; Merged: {i}")
    save_image_as_tiff(labels, os.path.join(root_dir, "LabelFreeCust", "val", f"pos{i}.tiff"))
