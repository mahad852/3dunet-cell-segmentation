python train.py \
    --mode "3d" \
    --output-model-path "models/seg_lfd.pth" \
    --train-ds-path "/home/mali2/datasets/CellSeg/LabelFreeCust/train" \
    --val-ds-path "/home/mali2/datasets/CellSeg/LabelFreeCust/val" \
    --epochs 50