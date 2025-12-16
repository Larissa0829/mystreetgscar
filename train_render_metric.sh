#!/bin/bash
scenes=("002") #  "002" "016" "021" "022" "090"
indexs=("0") # "0" "1"
for scene in "${scenes[@]}"; do
    for index in "${indexs[@]}"; do
        python train.py --config configs/example/waymo_train_${scene}_${index}.yaml
        python render.py --config configs/example/waymo_train_${scene}_${index}.yaml mode trajectory render_dir_name trajectory_right_rotate render_move_y 0 render_rotate_z 90
        python render.py --config configs/example/waymo_train_${scene}_${index}.yaml mode trajectory render_dir_name trajectory render_move_y 0 render_rotate_z 0
        python metrics.py --config configs/example/waymo_train_${scene}_${index}.yaml
        ## python metrics_mask.py --config configs/example/waymo_train_${scene}_${index}.yaml
    done
done
