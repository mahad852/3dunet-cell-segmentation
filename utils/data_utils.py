import h5py
from torch.utils.data import Dataset
from typing import List, Tuple
import os

from bioio import BioImage
import bioio_czi
import numpy as np
import cv2
import tifffile

from skimage.filters import threshold_otsu

def convert_torch_dataset_to_h5py(dataset: Dataset, keys: List[str], root_dir: str):
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    
    for i, data in enumerate(dataset):
        if len(data) != len(keys):
            raise ValueError("The number of keys for h5 dataset names should be the same as the number of data values, but found datavalues:", len(data), "Keys:", len(keys))
        
        h5_filename = f"file{i}.h5"
        h5_filepath = os.path.join(root_dir, h5_filename)


        with h5py.File(h5_filepath, "w") as h5file:
            for k_index, k in enumerate(keys):
                h5file.create_dataset(k, data=data[k_index])

def read_czi_image(img_path: str) -> np.ndarray:
    img = BioImage(img_path, reader=bioio_czi.Reader)
    return img.data[0]

def save_image_as_tiff(img: np.ndarray, fpath: str):
    tifffile.imwrite(fpath, img)

def get_mask_using_threshold(img: np.ndarray, channels: List[int]) -> np.ndarray:
    mask = np.zeros(shape=img.shape[1:])
    
    for i, channel in enumerate(channels):
        m = img[channel] >= threshold_otsu(img[channel])
        mask[m.nonzero()] = i + 1

    return mask    

def merge_channels(images: List[np.ndarray]):
    final_img = np.zeros((images[0].shape[0], len(images), images[0].shape[1], images[0].shape[2]))
    
    for i, img in enumerate(images):
        final_img[:, i, :, :] = img
    
    return final_img