#!/bin/bash
num_outputs=10
metric=ccc
save_path=$1

# Start evaluation
python main.py --modality="audio" \
               --root_dir=./ \
               --model_name=emo18 \
               --num_outputs=$num_outputs \
               --take_last_frame="true" \
               test  \
               --prediction_file=predictions.csv \
               --metric=$metric \
               --dataset_path=$save_path/data/test \
               --model_path=./training/model/best.pth.tar
