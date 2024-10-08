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
from datasets.CellDataset2D import CellDataset2D
from datasets.CellDatasetMIP import CellDatasetMIP
from datasets.AllenCellDataset import AllenCellDataset

import tifffile

class AddChannel(object):
    def __call__(self, arr):
        return np.expand_dims(arr, axis=0)


def main():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)


    val_imtrans = Compose([AddChannel()])
    val_segtrans = Compose([AddChannel()])

    # define image dataset, data loader
    ####################################### CUSTOM IMPL ###################################################
   
    val_ds = CellDataset2D(data_path='/home/mali2/datasets/CellSeg/Widefield Deconvolved', num_channels=2, transform_image=val_imtrans, transform_seg=val_segtrans, is_segmentation=True, is_train=False)
    # val_ds = CellDatasetMIP(data_path='/home/mali2/datasets/CellSeg/Widefield Deconvolved', num_channels=2, transform_image=val_imtrans, transform_seg=val_segtrans, is_segmentation=True)
    val_loader = DataLoader(val_ds, batch_size=2, num_workers=1, pin_memory=torch.cuda.is_available())

    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    iou_metric = MeanIoU(include_background=True, reduction="mean")
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    # create UNet, DiceLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    model.load_state_dict(torch.load('best_metric_model_segmentation2d_composite.pth', map_location=device, weights_only=True))
    # model.load_state_dict(torch.load('best_metric_model_segmentation2d_composite_nmip.pth', map_location=device, weights_only=True))
    
    writer = SummaryWriter()
    
    image_index = 0

    model.eval()
    with torch.no_grad():
        val_images = None
        val_labels = None
        val_outputs = None
        for val_data in val_loader:
            val_images, val_labels, image_paths = val_data[0].to(device), val_data[1].to(device), val_data[2]
            roi_size = (512, 512)
            sw_batch_size = 4
            val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
            val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]

            for label, output, img_path in zip(val_labels, val_outputs, image_paths):
                label, output = label.cpu().detach().numpy(), output.cpu().detach().numpy()

                output_fname = f"/home/mali2/datasets/CellSeg/generated/2d/seg/seg{img_path.split('/')[-1]}"
                output = output[0] * 255

                tifffile.imwrite(output_fname, output)
                image_index += 1

            # compute metric for current iteration
            dice_metric(y_pred=val_outputs, y=val_labels)
            iou_metric(y_pred=val_outputs, y=val_labels)
        # aggregate the final mean dice result
        metric = dice_metric.aggregate().item()
        val_iou = iou_metric.aggregate().item()

    
    print(f"Finished evaluation. IoU: {val_iou}. Dice: {metric}")


    writer.close()


if __name__ == "__main__":
    main()