# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import sys
from glob import glob

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# from torch.utils.data import DataLoader

import monai
from monai.data import ImageDataset, create_test_image_3d, decollate_batch, DataLoader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, MeanIoU
from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    RandRotate90,
    RandSpatialCrop,
    ScaleIntensity,
)
from monai.visualize import plot_2d_or_3d_image

from datasets.CellDataset import CellDataset
from datasets.AllenCellDataset import AllenCellDataset

import tifffile

from skimage.filters import threshold_otsu

class AddChannel(object):
    def __call__(self, arr):
        return np.expand_dims(arr, axis=0)

def get_mito_masks(imgs: np.ndarray):
    imgs = (imgs * 255)
    masks = np.zeros(shape=imgs.shape)
    for i, img in enumerate(imgs):
        for z in range(len(img[0])):
            masks[i][0][z] = img[0][z] >= threshold_otsu(img[0][z])
    return torch.Tensor(masks)

def denoise_img(img):
    return img * (img > 5)

def main():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)


    val_imtrans = Compose([AddChannel()])#, ScaleIntensity()])
    val_segtrans = Compose([AddChannel()])
   
    val_ds = CellDataset(data_path='/home/mali2/datasets/CellSeg/Widefield Deconvolved', num_channels=2, transform_image=val_imtrans, transform_seg=val_segtrans, is_segmentation=False)
    val_loader = DataLoader(val_ds, batch_size=4, num_workers=1, pin_memory=torch.cuda.is_available())

    iou_metric = MeanIoU(include_background=True, reduction="mean")

    loss_function = torch.nn.MSELoss()

    # create UNet, DiceLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = monai.networks.nets.UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    model.load_state_dict(torch.load('best_loss_model_regression_composite.pth', map_location=device, weights_only=True))
    
    model.eval()
    val_loss = 0
    vaL_samples = 0
    val_iou = 0
    with torch.no_grad():
        val_images = None
        val_labels = None
        val_outputs = None
        for val_data in val_loader:
            val_images, val_labels, img_pths = val_data[0].to(device), val_data[1].to(device), val_data[2]
            roi_size = (16, 512, 512)
            sw_batch_size = 4
            val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)

            val_loss += len(val_images) * loss_function(val_outputs.cpu(), val_labels.cpu())
            vaL_samples += len(val_images)

            for output, path in zip(val_outputs.detach().cpu(), img_pths):
                fname = path.split('/')[-1]
                out_file = f"/home/mali2/datasets/CellSeg/generated/{fname}"
                tifffile.imwrite(out_file, (output[0] * 255).cpu().numpy().astype(np.uint8))

            # print(val_masks, out_masks, val_masks.shape, out_masks.shape)

            val_masks = get_mito_masks(val_labels.detach().cpu().numpy())
            out_masks = get_mito_masks(val_outputs.detach().cpu().numpy())

            iou_metric(y_pred=out_masks, y=val_masks)

        # aggregate the final mean dice result
        val_loss = val_loss/vaL_samples
        val_iou = iou_metric.aggregate().item()
        # reset the status for next validation round

    print(f"Finished evaluation. val_loss: {val_loss} val_iou: {val_iou}")

if __name__ == "__main__":
    main()