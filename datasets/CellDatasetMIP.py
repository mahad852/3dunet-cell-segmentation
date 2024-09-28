from torch.utils.data import Dataset

from typing import List, Tuple

import os

import skimage.io

import numpy as np

from skimage.filters import threshold_otsu, threshold_sauvola, threshold_triangle

class CellDatasetMIP(Dataset):
    def __init__(self, 
                 data_path = '/home/mali2/datasets/CellSeg/Widefield Deconvolved Set 2',
                 num_channels = 2,
                 transform_image = None,
                 transform_seg = None,
                 is_segmentation=True,
                 img_depth = 26,
                 crop_depth = 16):

        self.root_folder = data_path        
        self.validate_path(self.root_folder, f"Incorrect data_path supplied. Expected a directory, {self.root_folder} is not a directory.")
        
        self.num_channels = num_channels
        self.transform_image = transform_image
        self.transform_seg = transform_seg
        self.is_segmentation = is_segmentation
        self.img_depth = img_depth
        self.crop_depth = crop_depth

        self.image_paths = self.get_image_paths(self.root_folder)

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
        mip = np.max(img, axis=0)
        return mip >= threshold_otsu(mip)
        
    def get_mito_mask(self, img: np.ndarray):
        mito_mip = np.max(img[1], axis=0)
        return mito_mip >= threshold_otsu(mito_mip)
        
    def get_tub_mask(self, img):
        tub_mip = np.max(img[0], axis=0)
        return tub_mip >= threshold_otsu(tub_mip)
    
    def get_labels(self, img):
        labels = np.zeros(shape = img.shape[-2:], dtype=np.int64)
        if self.num_channels == 1:
            mask = self.get_mask_for_single_channel_img(img)
            labels[mask.nonzero()] = 1
        else:
            tub_mask = self.get_tub_mask(img)
            mito_mask = self.get_mito_mask(img)
            labels[tub_mask.nonzero()] = 1
            labels[mito_mask.nonzero()] = 2
        return labels
    
    def normalize_img(self, img):
        return (img - img.mean())/img.std()
    
    def denoise_img(self, img):
        return img * (img > threshold_otsu(img))
    
    def scale_image(self, img):
        return img * (255/img.max())
    
    def convert_image_to_single_channel(self, img):
        img_cpy = np.zeros(img.shape)
        for c in range(len(img)):
            for z in range(len(img[c])):
                img_cpy[c][z] = self.scale_image(img[c][z])

        return img_cpy.mean(axis=0)
    
    def get_mito_image(self, img):
        return np.max(self.scale_image(img[1]), axis=0)
    
    def get_item_for_multichannel(self, img : np.ndarray):
        img = np.transpose(img, (1, 0, 2, 3)) # Z, C, H, W  ==> C, Z, H, W 
        labels = self.get_labels(img) == 2 if self.is_segmentation else (self.get_mito_image(img) / 255).astype(np.float32)
        return self.convert_image_to_single_channel(img) / 255, labels
    
    def get_item_for_single_channel(self, img : np.ndarray):
        img = self.scale_image(img) / 255
        labels = self.get_labels(img) if self.is_segmentation else np.max(img, axis=0).astype(np.float32)

        return self.denoise_img(img), labels

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, str]:
        img_path = self.image_paths[index]
        img = skimage.io.imread(img_path)

        img, labels = self.get_item_for_multichannel(img) if self.num_channels > 1 else self.get_item_for_single_channel(img)
        img = self.normalize_img(img)
        img = np.max(img, axis=0)

        if self.transform_image:
            img = self.transform_image(img)

        if self.transform_seg:
            labels = self.transform_seg(labels)

        return img.astype(np.float32), labels, self.image_paths[index]