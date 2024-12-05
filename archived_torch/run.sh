#!/bin/zsh

set -e

source ~/.zshrc

conda activate ml

train="True"
model_path="./experiments/CNN-DD_matrix-1000-B128-lr0.001/model.pth"

python train_model.py\
    --train $train\
    --trainer "unsupervised"\
    --model_type "CNN"\
    --optimizer_nm "Adam"\
    --loss_fn "ConvComplexLoss"\
    --data_dir "./data/DD_matrices.pt"\
    --data_name "DD_matrix"\
    --in_dim 1792\
    --out_dim 960\
    --num_epochs 1000\
    --batch_size 128\
    --learning_rate 0.001\
    --hidden_dim 256\
    --in_ch 1\
    --out_ch 1\
    --kernel_size 7\
    --model_path $model_path\
    --additional_info "IC"\
