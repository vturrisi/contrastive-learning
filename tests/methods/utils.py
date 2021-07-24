import numpy as np
import torch
from PIL import Image
from solo.utils.pretrain_dataloader import (
    dataset_with_index,
    prepare_dataloader,
    prepare_n_crop_transform,
    prepare_multicrop_transform,
    prepare_transform,
)
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FakeData

DATA_KWARGS = {
    "brightness": 0.4,
    "contrast": 0.4,
    "saturation": 0.2,
    "hue": 0.1,
    "gaussian_prob": 0.5,
    "solarization_prob": 0.5,
}


def gen_base_kwargs(cifar=False, momentum=False, multicrop=False, n_crops=2, n_small_crops=0):
    BASE_KWARGS = {
        "encoder": "resnet18",
        "n_classes": 10 if cifar else 100,
        "cifar": cifar,
        "zero_init_residual": True,
        "max_epochs": 2,
        "optimizer": "sgd",
        "lars": True,
        "lr": 0.3,
        "weight_decay": 1e-10,
        "classifier_lr": 0.5,
        "exclude_bias_n_norm": True,
        "accumulate_grad_batches": 1,
        "extra_optimizer_args": {"momentum": 0.9},
        "scheduler": "warmup_cosine",
        "min_lr": 0.0,
        "warmup_start_lr": 0.0,
        "warmup_epochs": 1,
        "multicrop": multicrop,
        "n_crops": n_crops,
        "n_small_crops": n_small_crops,
        "eta_lars": 0.02,
        "lr_decay_steps": None,
        "dali_device": "gpu",
        "last_batch_fill": False,
        "batch_size": 32,
        "num_workers": 4,
        "data_dir": "/data/datasets",
        "train_dir": "cifar10/train",
        "val_dir": "cifar10/val",
    }
    if momentum:
        BASE_KWARGS["base_tau_momentum"] = 0.99
        BASE_KWARGS["final_tau_momentum"] = 1.0
    return BASE_KWARGS


def gen_batch(b, n_classes, dataset):
    assert dataset in ["cifar10", "imagenet100"]

    if dataset == "cifar10":
        size = 32
    else:
        size = 224

    im = np.random.rand(size, size, 3) * 255
    im = Image.fromarray(im.astype("uint8")).convert("RGB")
    T = prepare_transform(dataset, multicrop=False, **DATA_KWARGS)
    T = prepare_n_crop_transform(T, n_crops=2)
    x1, x2 = T(im)
    x1 = x1.unsqueeze(0).repeat(b, 1, 1, 1).requires_grad_(True)
    x2 = x2.unsqueeze(0).repeat(b, 1, 1, 1).requires_grad_(True)

    idx = torch.arange(b)
    label = torch.randint(low=0, high=n_classes, size=(b,))

    batch, batch_idx = [idx, (x1, x2), label], 1

    return batch, batch_idx


def gen_classification_batch(b, n_classes, dataset):
    assert dataset in ["cifar10", "imagenet100"]

    if dataset == "cifar10":
        size = 32
    else:
        size = 224

    im = np.random.rand(size, size, 3) * 255
    im = Image.fromarray(im.astype("uint8")).convert("RGB")
    T = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
    )
    x = T(im)
    x = x.unsqueeze(0).repeat(b, 1, 1, 1).requires_grad_(True)

    label = torch.randint(low=0, high=n_classes, size=(b,))

    batch, batch_idx = (x, label), 1

    return batch, batch_idx


def prepare_dummy_dataloaders(
    dataset, n_crops, n_classes, multicrop=False, n_small_crops=0, batch_size=2
):
    T = prepare_transform(dataset, multicrop=multicrop, **DATA_KWARGS)
    if multicrop:
        size_crops = [224, 96] if dataset == "imagenet100" else [32, 24]
        T = prepare_multicrop_transform(T, size_crops=size_crops, n_crops=[n_crops, n_small_crops])
    else:
        T = prepare_n_crop_transform(T, n_crops)
    dataset = dataset_with_index(FakeData)(
        image_size=(3, 224, 224), num_classes=n_classes, transform=T
    )
    train_dl = prepare_dataloader(dataset, batch_size=batch_size, num_workers=0)

    # normal dataloader
    T_val = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
    )
    dataset = FakeData(image_size=(3, 224, 224), num_classes=n_classes, transform=T_val)
    val_dl = DataLoader(dataset, batch_size=batch_size, num_workers=0, drop_last=False)

    return train_dl, val_dl


def prepare_classification_dummy_dataloaders(dataset, n_classes):
    # normal dataloader
    T_val = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
    )
    dataset = FakeData(image_size=(3, 224, 224), num_classes=n_classes, transform=T_val)
    train_dl = val_dl = DataLoader(dataset, batch_size=2, num_workers=0, drop_last=False)

    return train_dl, val_dl
