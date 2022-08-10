python3 main_pretrain.py \
    --dataset imagenet100 \
    --backbone resnet18 \
    --train_data_path /datasets/imagenet-100/train \
    --val_data_path /datasets/imagenet-100/val \
    --max_epochs 400 \
    --devices 0,1 \
    --accelerator gpu \
    --strategy ddp \
    --sync_batchnorm \
    --precision 16 \
    --optimizer lars \
    --grad_clip_lars \
    --eta_lars 0.02 \
    --exclude_bias_n_norm_lars \
    --scheduler warmup_cosine \
    --lr 0.3 \
    --weight_decay 1e-4 \
    --batch_size 128 \
    --num_workers 4 \
    --brightness 0.8 \
    --contrast 0.8 \
    --saturation 0.8 \
    --hue 0.2 \
    --num_crops_per_aug 2 \
    --name simclr-400ep-imagenet100 \
    --data_format dali \
    --project solo-learn \
    --entity unitn-mhug \
    --wandb \
    --save_checkpoint \
    --method simclr \
    --temperature 0.2 \
    --proj_hidden_dim 2048
