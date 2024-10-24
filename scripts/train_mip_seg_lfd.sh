python train.py \
    --mode "2d" \
    --mip \
    --output-model-path "models/seg_lfd_mip.pth" \
    --train-ds-path "/home/mali2/datasets/CellSeg/LabelFreeCust/train" \
    --val-ds-path "/home/mali2/datasets/CellSeg/LabelFreeCust/val" \
    --epochs 50