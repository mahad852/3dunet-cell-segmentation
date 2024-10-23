python train.py \
    --mode "3d" \
    --input-model-path "models/seg_lfd.pth" \
    --output-model-path "models/finetuned_seg.pth" \
    --train-ds-path "/home/mali2/datasets/CellSeg/Widefield Deconvolved Set 2" \
    --val-ds-path "/home/mali2/datasets/CellSeg/Widefield Deconvolved" \
    --epochs 50 \

