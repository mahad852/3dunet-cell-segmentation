python train.py \
    --mode "2d" \
    --is-mip \
    --input-model-path "models/mip/segmentation/lfd.pth" \
    --output-model-path "models/mip/segmentation/finetuned.pth" \
    --train-ds-path "/home/mali2/datasets/CellSeg/Widefield Deconvolved Set 2" \
    --val-ds-path "/home/mali2/datasets/CellSeg/Widefield Deconvolved" \
    --epochs 300 \

