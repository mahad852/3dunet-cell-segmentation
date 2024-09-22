from torch.utils.data import Dataset
from typing import List
import os
import skimage.io
import numpy as np
from skimage.filters import threshold_otsu
import cv2

class AllenCellDataset(Dataset):
    def __init__(self, 
                 data_path = '/home/mali2/datasets/CellSeg/Widefield Deconvolved Set 2',
                 transform_image = None,
                 transform_seg = None,
                 is_train = True,
                 targets = ['mitochondria']):

        self.validate_path(data_path, f"Incorrect data_path supplied. Expected a directory, {data_path} is not a directory.")
        
        self.image_paths = self.get_image_paths(data_path, is_train)
        self.transform_image = transform_image
        self.transform_seg = transform_seg 
        self.targets = targets

    def __len__(self):
        return len(self.image_paths)
    
    def resize_image(self, img, width = 512, height=512):
        img_stack = np.zeros((len(img), width, height))

        for z in range(len(img)):
            img_stack[z, :, :] = cv2.resize(img[z, :, :], (width, height), interpolation=cv2.INTER_CUBIC)

        return img_stack
    
    def separate_channels(self, multi_channel_img):
        target_to_channel_index = {'mitochondria' : 14}

        transparent_light = multi_channel_img[:, 2, :, :]

        target_images = []

        for target in self.targets:
            target_images.append(multi_channel_img[:, target_to_channel_index[target], :, :])

        return transparent_light, target_images

    def get_image_paths(self, dir: str, is_train: bool) -> List[str]:
        image_paths = []
        for fname in os.listdir(dir):
            if fname.split('.')[-1] != 'tiff':
                continue 
            image_paths.append(os.path.join(dir, fname))

        if len(image_paths) == 0:
            raise ValueError(f"Expected tif files in the path: {dir}, but found none.")
        
        if is_train:
            return image_paths[:int(len(image_paths) * 0.70)]
        else:
            return image_paths[int(len(image_paths) * 0.70):]

    
    def read_image(self, image_path):
        return skimage.io.imread(image_path)
        
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
        
        mask = self.get_mask_for_single_channel_img(img)
        labels[mask.nonzero()] = 1

        return labels
    
    def normalize_img(self, img):
        return (img - img.mean())/img.std()
    
    def denoise_img(self, img):
        return img * (img > threshold_otsu(img))
    
    def __getitem__(self, index):        
        tlight, targets = self.separate_channels(self.read_image(image_path=self.image_paths[index]))

        tlight = self.resize_image(tlight)
        targets = [self.resize_image(target) for target in targets]

        tlight = self.normalize_img(self.denoise_img(tlight))   

        labels = [self.get_labels(target) for target in targets]

        if self.transform_image:
            tlight = self.transform_image(tlight)

        if self.transform_seg:
            labels = [self.transform_seg(label) for label in labels]

        return tlight, labels[0]