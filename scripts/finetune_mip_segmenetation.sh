python train.py \
    --mode "2d" \
    --is-mip \
    --input-model-path "models/seg_lfd_mip.pth" \
    --output-model-path "models/finetuned_seg_mip.pth" \
    --train-ds-path "/home/mali2/datasets/CellSeg/Widefield Deconvolved Set 2" \
    --val-ds-path "/home/mali2/datasets/CellSeg/Widefield Deconvolved" \
    --epochs 300 \

