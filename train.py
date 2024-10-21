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

from utils.arguments import get_train_args
from utils.arguments import get_mode, get_input_model_path, get_output_model_path, get_val_ds_path, get_train_ds_path, get_output_path, is_segmentation, is_mip

from utils.img_transformations import AddChannel

import monai.networks.nets
import monai.losses

import math

args = get_train_args()

mode = get_mode(args)
input_model_path = get_input_model_path(args)
output_model_path = get_output_model_path(args)

train_ds_path = get_train_ds_path(args)
val_ds_path = get_val_ds_path(args)
output_path = get_output_path(args)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = monai.networks.nets.UNet(
    spatial_dims=3 if mode == "3d" else 2,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device)

if input_model_path:
    model.load_state_dict(torch.load(input_model_path, weights_only=True))

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
    train_ds = CellDataset(data_path=train_ds_path, num_channels=2, transform_image=train_imtrans, transform_seg=train_segtrans, is_segmentation=is_segmentation(args), is_train=True)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=1, pin_memory=torch.cuda.is_available())

    val_ds = CellDataset(data_path=val_ds_path, num_channels=2, transform_image=val_imtrans, transform_seg=val_segtrans, is_segmentation=is_segmentation(args), is_train=False)
    val_loader = DataLoader(val_ds, batch_size=2, num_workers=1, pin_memory=torch.cuda.is_available())

elif is_mip(args):
    train_ds = CellDatasetMIP(data_path=train_ds_path, num_channels=2, transform_image=train_imtrans, transform_seg=train_segtrans, is_segmentation=is_segmentation(args))
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=1, pin_memory=torch.cuda.is_available())

    val_ds = CellDatasetMIP(data_path=val_ds_path, num_channels=2, transform_image=val_imtrans, transform_seg=val_segtrans, is_segmentation=is_segmentation(args))
    val_loader = DataLoader(val_ds, batch_size=2, num_workers=1, pin_memory=torch.cuda.is_available())
else:
    train_ds = CellDataset2D(data_path=train_ds_path, num_channels=2, transform_image=train_imtrans, transform_seg=train_segtrans, is_segmentation=is_segmentation(args), is_train=True)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=1, pin_memory=torch.cuda.is_available())

    val_ds = CellDataset2D(data_path=val_ds_path, num_channels=2, transform_image=val_imtrans, transform_seg=val_segtrans, is_segmentation=is_segmentation(args), is_train=False)
    val_loader = DataLoader(val_ds, batch_size=2, num_workers=1, pin_memory=torch.cuda.is_available())
 


dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
iou_metric = MeanIoU(include_background=True, reduction="mean")
post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

loss_function = monai.losses.DiceLoss(sigmoid=True) if is_segmentation(args) else torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), 1e-3)


num_epochs = 500
# start a typical PyTorch training
val_interval = 1
best_metric = -1
best_loss = 2 ** 31
best_metric_epoch = -1
epoch_loss_values = list()
metric_values = list()
writer = SummaryWriter()


for epoch in range(num_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{num_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_len = math.ceil(len(train_ds) / train_loader.batch_size)
        print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
        writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)

    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        val_loss = 0
        val_samples = 0
        with torch.no_grad():
            val_images = None
            val_labels = None
            val_outputs = None

            for val_data in val_loader:
                val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                roi_size = (16, 512, 512)
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

            if is_segmentation(args):
                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                val_iou = iou_metric.aggregate().item()

                # reset the status for next validation round
                dice_metric.reset()
                iou_metric.reset()

                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), output_model_path)
                    print("saved new best metric model")
                print(
                    "current epoch: {} current mean IoU: {:.4f} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                        epoch + 1, val_iou, metric, best_metric, best_metric_epoch
                    )
                )
            else:
                val_loss /= val_samples
                metric_values.append(val_loss)

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), output_model_path)
                    print("saved new best metric model")

                print(
                    "current epoch: {} current mean loss: {:.4f} current rmse: {:.4f} best mean loss: {:.4f} best rmse: {:.4f}; lr: {:.8f} at epoch {}".format(
                        epoch + 1, val_loss, np.sqrt(val_loss), best_loss, np.sqrt(best_loss), optimizer.param_groups[0]["lr"], best_metric_epoch
                    )
                )