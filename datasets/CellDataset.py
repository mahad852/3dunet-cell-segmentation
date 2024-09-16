from torch.utils.data import Dataset

from typing import List

import os

import skimage.io

import numpy as np

from skimage.filters import threshold_otsu

class CellDataset(Dataset):
    def __init__(self, 
                 data_path = '/home/mali2/datasets/CellSeg/Widefield Deconvolved Set 2',
                 num_channels = 2):

        self.root_folder = data_path        
        self.validate_path(self.root_folder, f"Incorrect data_path supplied. Expected a directory, {self.root_folder} is not a directory.")
        
        self.image_paths = self.get_image_paths(self.root_folder)
        self.num_channels = num_channels

    def __len__(self):
        return len(self.image_paths)

    def get_image_paths(self, dir: str) -> List[str]:
        image_paths = []
        for fname in os.listdir(dir):
            if fname.split('.')[-1] != 'tif':
                continue 
            image_paths.append(os.path.join(dir, fname))

        if len(image_paths) == 0:
            raise ValueError(f"Expected tif files in the path: {dir}, but found noen.")

        return image_paths
        
    def validate_path(self, path: str, error_msg: str):
        if not os.path.exists(path):
            raise ValueError(error_msg)
        
    def get_mask_for_single_channel_img(self, img):
        return img >= threshold_otsu(img)
        
    def get_mito_mask(self, img):
        return img[1] >= threshold_otsu(img[1])
        
    def get_tub_mask(self, img):
        return img[0] >= threshold_otsu(img[0])
    
    def get_labels(self, img):
        labels = np.zeros(shape = img.shape[-3:], dtype=np.long)
        if self.num_channels == 1:
            mask = self.get_mask_for_single_channel_img(img)
            labels[mask.nonzero()] = 1
        else:
            tub_mask = self.get_tub_mask(img)
            mito_mask = self.get_mito_mask(img)
            labels[tub_mask.nonzero()] = 1
            labels[mito_mask.nonzero()] = 2
        return labels
    
    def __getitem__(self, index):
        img_path = self.image_paths[index]
        
        img = skimage.io.imread(img_path)
        if self.num_channels > 1:
            img = np.transpose(img, (1, 0, 2, 3)) # Z, C, H, W  ==> C, Z, H, W 
        img = np.floor(img / 256) 
        return img, self.get_labels(img)