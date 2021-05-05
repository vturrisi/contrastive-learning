python3 ../main_contrastive.py \
    imagenet100 \
    resnet18 \
    --data_folder /datasets \
    --train_dir imagenet-100/train \
    --val_dir imagenet-100/test \
    --epochs 100 \
    --optimizer sgd \
    --lars \
    --scheduler warmup_cosine \
    --lr 0.3 \
    --weight_decay 1e-4 \
    --batch_size 128 \
    --temperature 0.2 \
    --gpus 0 1 \
    --num_workers 8 \
    --hidden_dim 2048 \
    --name simclr \
    --project contrastive_learning \
    --wandb

python3 ../main_contrastive.py \
    imagenet100 \
    resnet18 \
    --data_folder /datasets \
    --train_dir imagenet-100/train \
    --val_dir imagenet-100/test \
    --epochs 100 \
    --optimizer sgd \
    --lars \
    --scheduler warmup_cosine \
    --lr 0.3 \
    --weight_decay 1e-4 \
    --batch_size 128 \
    --temperature 0.2 \
    --gpus 0 1 \
    --num_workers 4 \
    --hidden_dim 2048 \
    --name simclr-dali \
    --dali \
    --project contrastive_learning \
    --wandb

python3 ../main_contrastive.py \
    imagenet100 \
    resnet18 \
    --data_folder /datasets \
    --train_dir imagenet-100/train \
    --val_dir imagenet-100/test \
    --epochs 100 \
    --optimizer sgd \
    --lars \
    --scheduler warmup_cosine \
    --lr 0.3 \
    --weight_decay 1e-4 \
    --batch_size 128 \
    --temperature 0.1 \
    --gpus 0 1 \
    --num_workers 4 \
    --hidden_dim 2048 \
    --multicrop \
    --n_crops 2 \
    --n_small_crops 6 \
    --name multi-crop-simclr-dali \
    --dali \
    --project contrastive_learning \
    --wandb

python3 ../main_contrastive.py \
    imagenet100 \
    resnet18 \
    --data_folder /datasets \
    --train_dir imagenet-100/train \
    --val_dir imagenet-100/test \
    --epochs 100 \
    --optimizer sgd \
    --lars \
    --scheduler warmup_cosine \
    --lr 0.3 \
    --weight_decay 1e-4 \
    --batch_size 128 \
    --temperature 0.1 \
    --gpus 0 1 \
    --num_workers 4 \
    --hidden_dim 2048 \
    --multicrop \
    --supervised \
    --n_crops 2 \
    --n_small_crops 6 \
    --name multi-crop-supervised-simclr-dali \
    --dali \
    --project contrastive_learning \
    --wandb
