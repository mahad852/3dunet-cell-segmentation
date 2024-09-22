import skimage
import numpy as np
import os
import cv2

root_dir = "/Users/mahad/Downloads"
mito_dir = os.path.join(root_dir, "mitochondria")
tlight_dir = os.path.join(root_dir, "transparent_light")

images_path = root_dir

if not os.path.exists(images_path):
    os.makedirs(images_path)

if not os.path.exists(mito_dir):
    os.makedirs(mito_dir)

if not os.path.exists(tlight_dir):
    os.makedirs(tlight_dir)


def separate_channels(multi_channel_img):
    transparent_light = multi_channel_img[:, 2, :, :]
    mitochondria = multi_channel_img[:, 14, :, :]

    return transparent_light, mitochondria

def resize_image(img, width = 512, height=512):
    img_stack = np.zeros((len(img), 512, 512))

    for z in range(len(img)):
        img_stack[z, :, :] = cv2.resize(img[z, :, :], (width, height), interpolation=cv2.INTER_CUBIC)

    return img_stack

def read_image(image_path):
    return skimage.io.imread(image_path)


image_fname = "1018.tiff"

tlight, mito = separate_channels(read_image(os.path.join(images_path, image_fname)))
mito = resize_image(mito)
tlight = resize_image(tlight)

mito_impath = os.path.join(mito_dir, image_fname)
tlight_impath = os.path.join(tlight_dir, image_fname)

skimage.io.imsave(mito_impath, mito)
skimage.io.imsave(tlight_impath, tlight)