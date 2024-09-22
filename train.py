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

class AddChannel(object):
    def __call__(self, arr):
        return np.expand_dims(arr, axis=0)


def main():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # define transforms for image and segmentation
    train_imtrans = Compose(
        [
            AddChannel(),
            ScaleIntensity(),
            RandSpatialCrop((16, 512, 512), random_size=False, random_center=False),
            RandRotate90(prob=0.5, spatial_axes=(1, 2)),
        ]
    )
    train_segtrans = Compose(
        [
            AddChannel(),
            RandSpatialCrop((16, 512, 512), random_size=False, random_center=False),
            RandRotate90(prob=0.5, spatial_axes=(1, 2)),
        ]
    )

    val_imtrans = Compose([AddChannel(), ScaleIntensity()])
    val_segtrans = Compose([AddChannel()])

    # define image dataset, data loader
    ####################################### CUSTOM IMPL ###################################################
   
    # check_ds = CellDataset(data_path='/home/mali2/datasets/CellSeg/Widefield Deconvolved Set 2/Mitochondria Channel', num_channels=1)
    check_ds = AllenCellDataset(data_path='/home/mali2/datasets/CellSeg/AllenCellData', transform_image=train_imtrans, transform_seg=train_segtrans, is_train=True)
    check_loader = DataLoader(check_ds, batch_size=2, num_workers=1, pin_memory=torch.cuda.is_available())
    im, seg = monai.utils.misc.first(check_loader)
    print(im.shape, seg.shape)

    # create a training data loader
    # train_ds = CellDataset(data_path='/home/mali2/datasets/CellSeg/Widefield Deconvolved Set 2/Mitochondria Channel', num_channels=1, transform_image=train_imtrans, transform_seg=train_segtrans)
    train_ds = AllenCellDataset(data_path='/home/mali2/datasets/CellSeg/AllenCellData', transform_image=train_imtrans, transform_seg=train_segtrans, is_train=True)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=1, pin_memory=torch.cuda.is_available())

    # val_ds = CellDataset(data_path='/home/mali2/datasets/CellSeg/Widefield Deconvolved/Mitochondria Channel', num_channels=1, transform_image=val_imtrans, transform_seg=val_segtrans)
    val_ds = AllenCellDataset(data_path='/home/mali2/datasets/CellSeg/AllenCellData', transform_image=val_imtrans, transform_seg=val_segtrans, is_train=False)
    val_loader = DataLoader(val_ds, batch_size=2, num_workers=1, pin_memory=torch.cuda.is_available())


    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    iou_metric = MeanIoU(include_background=True, reduction="mean")
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

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
    loss_function = monai.losses.DiceLoss(sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-2)

    num_epochs = 20
    # start a typical PyTorch training
    val_interval = 1
    best_metric = -1
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
            # print(inputs.shape, labels.shape)
            outputs = model(inputs)
            # print(inputs.shape, labels.shape, outputs.shape, labels, outputs)

            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_images = None
                val_labels = None
                val_outputs = None
                for val_data in val_loader:
                    val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                    roi_size = (16, 512, 512)
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
                    val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]

                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels)
                    iou_metric(y_pred=val_outputs, y=val_labels)
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
                    torch.save(model.state_dict(), "best_metric_model_segmentation3d_array.pth")
                    print("saved new best metric model")
                print(
                    "current epoch: {} current mean IoU: {:.4f} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                        epoch + 1, val_iou, metric, best_metric, best_metric_epoch
                    )
                )
                writer.add_scalar("val_mean_dice", metric, epoch + 1)
                # plot the last model output as GIF image in TensorBoard with the corresponding image and label
                # plot_2d_or_3d_image(val_images, epoch + 1, writer, index=0, tag="image")
                # plot_2d_or_3d_image(val_labels, epoch + 1, writer, index=0, tag="label")
                # plot_2d_or_3d_image(val_outputs, epoch + 1, writer, index=0, tag="output")

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()


if __name__ == "__main__":
    main()