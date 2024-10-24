python train.py \
    --mode "2d" \
    --is-mip \
    --output-model-path "models/mip/segmentation/lfd.pth" \
    --train-ds-path "/home/mali2/datasets/CellSeg/LabelFreeCust/train" \
    --val-ds-path "/home/mali2/datasets/CellSeg/LabelFreeCust/val" \
    --epochs 50