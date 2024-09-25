from torch.utils.data import Dataset

from typing import List

import os

import skimage.io

import numpy as np

from skimage.filters import threshold_otsu, threshold_sauvola, threshold_triangle

class CellDataset(Dataset):
    def __init__(self, 
                 data_path = '/home/mali2/datasets/CellSeg/Widefield Deconvolved Set 2',
                 num_channels = 2,
                 transform_image = None,
                 transform_seg = None,
                 is_segmentation=True,
                 is_train = True,
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
        self.is_train = is_train

        self.image_paths = self.get_image_paths(self.root_folder)
        self.slices = self.get_slices(self.image_paths)

    def __len__(self):
        return len(self.image_paths)
    
    def get_slices(self, img_paths):
        if self.is_train:
            slices = [slice_start for slice_start in range(self.img_depth - self.crop_depth + 1)] * len(img_paths)
        else:
            slices = [0] * len(img_paths)
        return slices

    def get_image_paths(self, dir: str) -> List[str]:
        image_paths = []
        for fname in os.listdir(dir):
            if fname.split('.')[-1] != 'tif':
                continue 
            if self.is_train:
                image_paths.extend([os.path.join(dir, fname)] * (self.img_depth - self.crop_depth + 1))
            else:
                image_paths.append(os.path.join(dir, fname))

        if len(image_paths) == 0:
            raise ValueError(f"Expected tif files in the path: {dir}, but found noen.")

        return image_paths
        
    def validate_path(self, path: str, error_msg: str):
        if not os.path.exists(path):
            raise ValueError(error_msg)
        
    def get_mask_for_single_channel_img(self, img):
        return img >= threshold_otsu(img)
        
    def get_mito_mask(self, img: np.ndarray):
        return img[1] >= threshold_otsu(img[1])
        
    def get_tub_mask(self, img):
        return img[0] >= threshold_otsu(img[0])
    
    def get_labels(self, img):
        labels = np.zeros(shape = img.shape[-3:], dtype=np.int64)
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
            img_cpy[c] = self.scale_image(img[c])

        return img_cpy.mean(axis=0)
    
    def get_mito_image(self, img):
        return self.scale_image(img[1])
    
    def get_item_for_multichannel(self, img : np.ndarray):
        img = np.transpose(img, (1, 0, 2, 3)) # Z, C, H, W  ==> C, Z, H, W 
        labels = self.get_labels(img) == 2 if self.is_segmentation else (self.get_mito_image(img) / 255).astype(np.float32)
        return self.convert_image_to_single_channel(img) / 255, labels
    
    def get_item_for_single_channel(self, img : np.ndarray):
        return self.denoise_img(img), self.get_labels(img)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        img = skimage.io.imread(img_path)[self.slices[index] : self.slices[index] + (self.crop_depth if self.is_train else self.img_depth)]

        img, labels = self.get_item_for_multichannel(img) if self.num_channels > 1 else self.get_item_for_single_channel(img)
        img = self.normalize_img(img)

        if self.transform_image:
            img = self.transform_image(img)

        if self.transform_seg:
            labels = self.transform_seg(labels)

        return img.astype(np.float32), labels, self.image_paths[index]