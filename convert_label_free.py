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


ds = AllenCellDataset(os.path.join(root_dir, "AllenCellData"), targets=["microtubule", "mitochondria"], transform_image=transform_img, transform_seg=transform_img)

for i, (imputs, labels) in enumerate(ds):
    labels = merge_channels(labels)
    print(f"Merged: {i}")
    save_image_as_tiff(labels, os.path.join(root_dir, "LabelFreeCust", f"pos{i}.tiff"))
