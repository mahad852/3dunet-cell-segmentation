import numpy as np
from utils.data_utils import read_czi_image, get_mask_using_threshold
from torch.utils.data import Dataset
from typing import List, Tuple
import os

class CZIDataset(Dataset):
    def __init__(self, root_path: str, input_channel: int, output_channels: List[int], transform_img = None, transform_seg = None) -> None:
        super().__init__()
        self.root_path = root_path
        self.image_paths = self.get_image_paths()
        self.input_channel = input_channel
        self.output_channels = output_channels
        self.transform_img = transform_img
        self.transform_seg = transform_seg

    def get_image_paths(self) -> List[str]:
        image_paths = []
        for fname in os.listdir(self.root_path):
            if "czi" not in fname:
                continue
            image_paths.append(os.path.join(self.root_path, fname))
        return image_paths
    
    def __getitem__(self, index) -> Tuple[np.ndarray, np.ndarray]:
        img = read_czi_image(self.image_paths[index])    

        input = img[self.input_channel]
        if self.transform_img:
            input = self.transform_img(input)
        
        mask = get_mask_using_threshold(img, channels=self.output_channels)
        if self.transform_seg:
            mask = self.transform_seg(mask)

        return input, mask