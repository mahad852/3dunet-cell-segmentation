import numpy as np
import cv2
from typing import List, Tuple

class AddChannel(object):
    def __call__(self, img: np.ndarray) -> np.ndarray:
        return np.expand_dims(img, axis=0)

class ResizeImage(object):
    def __init__(self, size: Tuple[int]) -> None:
        self.size = size
    def __call__(self, img: np.ndarray) -> np.ndarray:    
        img_stack = np.zeros((len(img), *self.size))

        for z in range(len(img)):
            img_stack[z, :, :] = cv2.resize(img[z, :, :], self.size, interpolation=cv2.INTER_CUBIC)

        return img_stack
    
class CropImageDepth(object):
    def __init__(self, depth: int, center = None):
        self.depth = depth
        self.center = center

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if self.center == None:
            self.center = int(img.shape[0] / 2)
        
        start = self.center - int(self.depth / 2)
        end = start + self.depth

        return img[start:end]