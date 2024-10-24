python train.py \
    --mode "3d" \
    --input-model-path "models/3d/segmentation/lfd.pth" \
    --output-model-path "models/3d/segmentation/finetuned.pth" \
    --train-ds-path "/home/mali2/datasets/CellSeg/Widefield Deconvolved Set 2" \
    --val-ds-path "/home/mali2/datasets/CellSeg/Widefield Deconvolved" \
    --epochs 50 \

