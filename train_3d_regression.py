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
            # ScaleIntensity(),
            RandSpatialCrop((16, 512, 512), random_size=False),
            # RandRotate90(prob=0.5, spatial_axes=(1, 2)),
        ]
    )
    train_segtrans = Compose(
        [
            AddChannel(),
            RandSpatialCrop((16, 512, 512), random_size=False),
            # RandRotate90(prob=0.5, spatial_axes=(1, 2)),
        ]
    )

    val_imtrans = Compose([AddChannel()])#, ScaleIntensity()])
    val_segtrans = Compose([AddChannel()])

    # define image dataset, data loader
    ####################################### CUSTOM IMPL ###################################################
   
    check_ds = CellDataset(data_path='/home/mali2/datasets/CellSeg/Widefield Deconvolved Set 2', num_channels=2, is_segmentation=False)
    # check_ds = AllenCellDataset(data_path='/home/mali2/datasets/CellSeg/AllenCellData', transform_image=train_imtrans, transform_seg=train_segtrans, is_train=True)
    check_loader = DataLoader(check_ds, batch_size=4, num_workers=1, pin_memory=torch.cuda.is_available())
    im, seg, _ = monai.utils.misc.first(check_loader)
    print(im.shape, seg.shape)

    # create a training data loader
    train_ds = CellDataset(data_path='/home/mali2/datasets/CellSeg/Widefield Deconvolved Set 2', num_channels=2, transform_image=train_imtrans, transform_seg=train_segtrans, is_segmentation=False)
    # train_ds = AllenCellDataset(data_path='/home/mali2/datasets/CellSeg/AllenCellData', transform_image=train_imtrans, transform_seg=train_segtrans, is_train=True)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=1, pin_memory=torch.cuda.is_available())

    val_ds = CellDataset(data_path='/home/mali2/datasets/CellSeg/Widefield Deconvolved', num_channels=2, transform_image=val_imtrans, transform_seg=val_segtrans, is_segmentation=False)
    # val_ds = AllenCellDataset(data_path='/home/mali2/datasets/CellSeg/AllenCellData', transform_image=val_imtrans, transform_seg=val_segtrans, is_train=False)
    val_loader = DataLoader(val_ds, batch_size=2, num_workers=1, pin_memory=torch.cuda.is_available())

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
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)

    num_epochs = 10000
    # start a typical PyTorch training
    val_interval = 1
    best_loss = 2 ** 31
    best_loss_epoch = -1
    epoch_loss_values = list()
    loss_values = list()
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
            val_loss = 0
            vaL_samples = 0
            with torch.no_grad():
                val_images = None
                val_labels = None
                val_outputs = None
                for val_data in val_loader:
                    val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                    roi_size = (16, 512, 512)
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)

                    val_loss += len(val_images) * loss_function(val_outputs.cpu(), val_labels.cpu())
                    vaL_samples += len(val_images)
                    
                # aggregate the final mean dice result
                val_loss = val_loss/vaL_samples
                # reset the status for next validation round

                loss_values.append(val_loss)
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_loss_epoch = epoch + 1
                    torch.save(model.state_dict(), "best_loss_model_regression_composite.pth")
                    print("saved new best metric model")
                print(
                    "current epoch: {} current mean loss: {:.4f} current rmse: {:.4f} best mean loss: {:.4f} best rmse: {:.4f}; lr: {:.8f} at epoch {}".format(
                        epoch + 1, val_loss, np.sqrt(val_loss), best_loss, np.sqrt(best_loss), best_loss_epoch, optimizer.param_groups[0]["lr"]
                    )
                )
                writer.add_scalar("val_mean_loss", val_loss, epoch + 1)
                # plot the last model output as GIF image in TensorBoard with the corresponding image and label
                # plot_2d_or_3d_image(val_images, epoch + 1, writer, index=0, tag="image")
                # plot_2d_or_3d_image(val_labels, epoch + 1, writer, index=0, tag="label")
                # plot_2d_or_3d_image(val_outputs, epoch + 1, writer, index=0, tag="output")

    print(f"train completed, best_metric: {best_loss:.4f} at epoch: {best_loss_epoch}")
    writer.close()


if __name__ == "__main__":
    main()