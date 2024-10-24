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

import monai
from monai.data import decollate_batch, DataLoader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, MeanIoU
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
)
from datasets.CellDataset import CellDataset
from datasets.CellDataset2D import CellDataset2D
from datasets.CellDatasetMIP import CellDatasetMIP

from datasets.AllenCellDataset import AllenCellDataset

from utils.arguments import get_test_args
from utils.arguments import get_mode, get_input_model_path, get_val_ds_path, get_output_path, is_segmentation, is_mip

from utils.data_utils import get_mask_using_threshold, scale_image

from utils.img_transformations import AddChannel

import monai.networks.nets
import monai.losses

import tifffile

import os

args = get_test_args()

mode = get_mode(args)
input_model_path = get_input_model_path(args)

val_ds_path = get_val_ds_path(args)
output_path = get_output_path(args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("*" * 10, "Preparing Model...", "*" * 10)

model = monai.networks.nets.UNet(
    spatial_dims=3 if mode == "3d" else 2,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device)

model.load_state_dict(torch.load(input_model_path, weights_only=True))

if not os.path.exists(output_path):
    os.makedirs(output_path)

train_imtrans = Compose(
    [
        AddChannel(),
    ]
)

train_segtrans = Compose(
    [
        AddChannel(),
    ]
)

val_imtrans = Compose([AddChannel()])#, ScaleIntensity()])
val_segtrans = Compose([AddChannel()])

if mode == "3d":
    val_ds = CellDataset(data_path=val_ds_path, num_channels=2, transform_image=val_imtrans, transform_seg=val_segtrans, is_segmentation=is_segmentation(args), is_train=False)
    val_loader = DataLoader(val_ds, batch_size=2, num_workers=1, pin_memory=torch.cuda.is_available())

elif is_mip(args):
    val_ds = CellDatasetMIP(data_path=val_ds_path, num_channels=2, transform_image=val_imtrans, transform_seg=val_segtrans, is_segmentation=is_segmentation(args))
    val_loader = DataLoader(val_ds, batch_size=2, num_workers=1, pin_memory=torch.cuda.is_available())
else:
    val_ds = CellDataset2D(data_path=val_ds_path, num_channels=2, transform_image=val_imtrans, transform_seg=val_segtrans, is_segmentation=is_segmentation(args), is_train=False)
    val_loader = DataLoader(val_ds, batch_size=2, num_workers=1, pin_memory=torch.cuda.is_available())
 


dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
iou_metric = MeanIoU(include_background=True, reduction="mean")
post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

loss_function = monai.losses.DiceLoss(sigmoid=True) if is_segmentation(args) else torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), 1e-3)


writer = SummaryWriter()

print("*" * 10, "Beginning testing", "*" * 10)

model.eval()
val_loss = 0
val_samples = 0
with torch.no_grad():
    val_images = None
    val_labels = None
    val_outputs = None

    for val_data in val_loader:
        val_images, val_labels, image_paths = val_data[0].to(device), val_data[1].to(device), val_data[3]
        roi_size = (16, 512, 512) if mode == "3d" else (512, 512)
        sw_batch_size = 4
        val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
        if is_segmentation(args):
            val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]            
            # compute metric for current iteration
            dice_metric(y_pred=val_outputs, y=val_labels)
            iou_metric(y_pred=val_outputs, y=val_labels)
        else:
            val_loss += len(val_images) * loss_function(val_outputs.cpu(), val_labels.cpu())
            val_samples += len(val_images)

            val_masks = get_mask_using_threshold(val_labels.detach().cpu().numpy(), channels=[0])
            out_masks = get_mask_using_threshold(val_outputs.detach().cpu().numpy(), channels=[0])
            iou_metric(y_pred=out_masks, y=val_masks)
        
        for label, output, img_path in zip(val_labels, val_outputs, image_paths):
            label, output = label.cpu().detach().numpy(), output.cpu().detach().numpy()

            if is_segmentation(args):
                tifffile.imwrite(os.path.join(output_path, f"seg{img_path.split('/')[-1]}"), output[0] * 255)
            else:
                tifffile.imwrite(os.path.join(output_path, f"reg{img_path.split('/')[-1]}"), (scale_image(output[0]).cpu().numpy()).astype(np.uint16))

if is_segmentation(args):
    metric = dice_metric.aggregate().item()
    val_iou = iou_metric.aggregate().item()
    print(f"Finished evaluation. IoU: {val_iou}. Dice: {metric}")
else:
    val_iou = iou_metric.aggregate().item()
    val_loss /= val_samples
    print(f"Finished evaluation. val_loss: {val_loss} val_iou: {val_iou}")
