from torch.utils.data import Dataset

from typing import List, Tuple

import os

import skimage.io

import numpy as np

from skimage.filters import threshold_otsu, threshold_sauvola, threshold_triangle

class CellDataset2D(Dataset):
    def __init__(self, 
                 data_path = '/home/mali2/datasets/CellSeg/Widefield Deconvolved Set 2',
                 num_channels = 2,
                 transform_image = None,
                 transform_seg = None,
                 is_segmentation=True,
                 is_train = True,
                 img_depth = 26):

        self.root_folder = data_path        
        self.validate_path(self.root_folder, f"Incorrect data_path supplied. Expected a directory, {self.root_folder} is not a directory.")
        
        self.num_channels = num_channels
        self.transform_image = transform_image
        self.transform_seg = transform_seg
        self.is_segmentation = is_segmentation
        self.img_depth = img_depth
        self.is_train = is_train

        self.image_paths = self.get_image_paths(self.root_folder)
        self.zs = self.get_zs(self.image_paths)

    def __len__(self):
        return len(self.image_paths)
    
    def get_zs(self, img_paths) -> List[int]:
        return [i % self.img_depth for i, _ in enumerate(img_paths)]
            
    def get_image_paths(self, dir: str) -> List[str]:
        image_paths = []
        for fname in os.listdir(dir):
            if fname.split('.')[-1] != 'tif':
                continue 
            image_paths.extend([os.path.join(dir, fname)] * self.img_depth)

        if len(image_paths) == 0:
            raise ValueError(f"Expected tif files in the path: {dir}, but found noen.")

        return image_paths
        
    def validate_path(self, path: str, error_msg: str):
        if not os.path.exists(path):
            raise ValueError(error_msg)
        
    def get_mask_for_single_channel_img(self, img: np.ndarray) -> np.ndarray:
        return img >= threshold_otsu(img)
        
    def get_mito_mask(self, img: np.ndarray) -> np.ndarray:
        return img[1] >= threshold_otsu(img[1])
        
    def get_tub_mask(self, img) -> np.ndarray:
        return img[0] >= threshold_otsu(img[0])
    
    def get_labels(self, img: np.ndarray) -> np.ndarray:
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
    
    def normalize_img(self, img: np.ndarray) -> np.ndarray:
        return (img - img.mean())/img.std()
    
    def denoise_img(self, img: np.ndarray) -> np.ndarray:
        return img * (img > threshold_otsu(img))
    
    def scale_image(self, img: np.ndarray) -> np.ndarray:
        return ((img - img.min())/(img.max() - img.min())) * 255
    
    def convert_image_to_single_channel(self, img: np.ndarray) -> np.ndarray:
        img_cpy = np.zeros(img.shape)
        for c in range(len(img_cpy)):
            img_cpy[c] = self.scale_image(img[c])

        return img_cpy.mean(axis=0)
                
    def get_mito_image(self, img: np.ndarray) -> np.ndarray:
        return self.scale_image(img[1])
    
    def get_item_for_multichannel(self, img : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        img = np.transpose(img, (1, 0, 2, 3)) # Z, C, H, W  ==> C, Z, H, W 
        labels = self.get_labels(img) == 2 if self.is_segmentation else self.denoise_img(self.get_mito_image(img) / 255).astype(np.float32)
        return self.convert_image_to_single_channel(img)/255, labels
    
    def get_item_for_single_channel(self, img : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        img = self.scale_image(img) / 255
        labels = self.get_labels(img) if self.is_segmentation else img.astype(np.float32)

        return self.denoise_img(img), labels

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        img_path = self.image_paths[index]
        img = skimage.io.imread(img_path)

        img, labels = self.get_item_for_multichannel(img) if self.num_channels > 1 else self.get_item_for_single_channel(img)
        img = img[self.zs[index]]
        labels = labels[self.zs[index]]

        if self.transform_image:
            img = self.transform_image(img)

        if self.transform_seg:
            labels = self.transform_seg(labels)

        return img.astype(np.float32), labels, self.image_paths[index]