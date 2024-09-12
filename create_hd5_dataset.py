from utils.data_utils import convert_torch_dataset_to_h5py
from datasets.CellDataset import CellDataset

# ds = CellDataset(data_path='/Users/mahad/Downloads/Widefield Deconvolved Set 2')
ds = CellDataset(data_path='/home/mali2/datasets/CellSeg/Widefield Deconvolved Set 2')
convert_torch_dataset_to_h5py(ds, ["raw", "label"], root_dir="/home/mali2/datasets/CellSeg/train/hd5_files_cell")

ds = CellDataset(data_path='/home/mali2/datasets/CellSeg/Widefield Deconvolved')
convert_torch_dataset_to_h5py(ds, ["raw", "label"], root_dir="/home/mali2/datasets/CellSeg/val/hd5_files_cell")