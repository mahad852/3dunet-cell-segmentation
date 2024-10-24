python train.py \
    --mode "2d" \
    --output-model-path "models/2d/segmentation/lfd.pth" \
    --train-ds-path "/home/mali2/datasets/CellSeg/LabelFreeCust/train" \
    --val-ds-path "/home/mali2/datasets/CellSeg/LabelFreeCust/val" \
    --epochs 10