python train.py \
    --mode "2d" \
    --input-model-path "models/2d/segmentation/lfd.pth" \
    --output-model-path "models/2d/segmentation/finetuned.pth" \
    --train-ds-path "/home/mali2/datasets/CellSeg/Widefield Deconvolved Set 2" \
    --val-ds-path "/home/mali2/datasets/CellSeg/Widefield Deconvolved" \
    --epochs 100 \

