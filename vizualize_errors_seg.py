import numpy as np
import tifffile
from utils.data_utils import save_image_as_tiff, get_mask_using_threshold
import re
import os

from sklearn.metrics import confusion_matrix  

def compute_iou(y_pred, y_true):
    # ytrue, ypred is a flatten vector
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    current = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # compute mean iou
    intersection = np.diag(current)
    ground_truth_set = current.sum(axis=1)
    predicted_set = current.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    IoU = intersection / union.astype(np.float32)
    return np.mean(IoU)

path1 = "/Users/mahad/samples/generated/seg"
path2 = "/Users/mahad/samples/Widefield Deconvolved/Mitochondria Channel"

p1_map = {}
for fname in os.listdir(path1):
    p1_map[re.findall(r"\d+", fname)[0]] = fname

p2_map = {}
for fname in os.listdir(path2):
    p2_map[re.findall(r"\d+", fname)[0]] = fname


ious = []

for i in p1_map.keys():
    if i not in p2_map:
        continue
    
    p2_f = tifffile.imread(os.path.join(path2, p2_map[i]))
    
    mask2 = get_mask_using_threshold(np.expand_dims(p2_f, axis=0), channels=[0]) > 0
    mask1 = tifffile.imread(os.path.join(path1, p1_map[i])) > 0

    total_elems = mask1.shape[0] * mask1.shape[1] * mask1.shape[2]

    mask_error = (mask1 == mask2)

    print((mask1 * mask2).sum()/(mask1.sum() + mask2.sum() - (mask1 * mask2).sum()))

    print(p2_map[i], p1_map[i])

    true_negatives = np.logical_and(np.logical_not(mask1), np.logical_not(mask2)).sum()
    true_positives = np.logical_and(mask1, mask2)

    iou = compute_iou(mask1, mask2)
    ious.append(iou)

    print("IOU:", iou)


    save_image_as_tiff((mask_error * 255).astype(np.uint8), os.path.join(f"/Users/mahad/Downloads/genSegErrors/pos{i}.tif"))
    save_image_as_tiff((mask1 * 255).astype(np.uint8), os.path.join(f"/Users/mahad/Downloads/genSegErrors/maskpos{i}.tif"))

print(np.array(ious).mean())