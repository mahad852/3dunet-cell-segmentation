python train.py \
    --mode "3d" \
    --output-model-path "models/3d/segmentation/lfd.pth" \
    --train-ds-path "/home/mali2/datasets/CellSeg/LabelFreeCust/train" \
    --val-ds-path "/home/mali2/datasets/CellSeg/LabelFreeCust/val" \
    --epochs 50