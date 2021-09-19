import argparse

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from solo.methods import DINO

from .utils import DATA_KWARGS, gen_base_kwargs, gen_batch, prepare_dummy_dataloaders


def test_dino():
    method_kwargs = {
        "proj_output_dim": 256,
        "proj_hidden_dim": 2048,
        "num_prototypes": 4096,
        "norm_last_layer": True,
        "clip_grad": 0,
        "freeze_last_layer": 1,
        "student_temperature": 0.1,
        "teacher_temperature": 0.7,
        "warmup_teacher_temperature": 0.4,
        "warmup_teacher_temperature_epochs": 1,
        "momentum_classifier": True,
        "use_bn_in_head": False,
    }

    BASE_KWARGS = gen_base_kwargs(cifar=False, momentum=True, batch_size=2)
    kwargs = {**BASE_KWARGS, **DATA_KWARGS, **method_kwargs}
    model = DINO(**kwargs, online_knn_eval=False)

    # test arguments
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    assert model.add_model_specific_args(parser) is not None

    # test parameters
    assert model.learnable_params is not None

    # test forward
    batch, _ = gen_batch(BASE_KWARGS["batch_size"], BASE_KWARGS["num_classes"], "imagenet100")
    out = model(batch[1][0])
    assert (
        "logits" in out
        and isinstance(out["logits"], torch.Tensor)
        and out["logits"].size() == (BASE_KWARGS["batch_size"], BASE_KWARGS["num_classes"])
    )
    assert (
        "feats" in out
        and isinstance(out["feats"], torch.Tensor)
        and out["feats"].size() == (BASE_KWARGS["batch_size"], model.features_dim)
    )
    assert (
        "p" in out
        and isinstance(out["p"], torch.Tensor)
        and out["p"].size() == (BASE_KWARGS["batch_size"], method_kwargs["num_prototypes"])
    )

    # imagenet
    BASE_KWARGS = gen_base_kwargs(cifar=False, momentum=True, batch_size=2)
    kwargs = {**BASE_KWARGS, **DATA_KWARGS, **method_kwargs}
    model = DINO(**kwargs, online_knn_eval=False)

    args = argparse.Namespace(**kwargs)
    trainer = Trainer.from_argparse_args(args, fast_dev_run=True)
    train_dl, val_dl = prepare_dummy_dataloaders(
        "imagenet100",
        num_crops=BASE_KWARGS["num_crops"],
        num_small_crops=0,
        num_classes=BASE_KWARGS["num_classes"],
        multicrop=False,
        batch_size=BASE_KWARGS["batch_size"],
    )
    trainer.fit(model, train_dl, val_dl)

    # cifar
    BASE_KWARGS = gen_base_kwargs(cifar=True, momentum=True, batch_size=2)
    kwargs = {**BASE_KWARGS, **DATA_KWARGS, **method_kwargs}
    model = DINO(**kwargs, online_knn_eval=False)

    args = argparse.Namespace(**kwargs)
    trainer = Trainer.from_argparse_args(args, fast_dev_run=True)
    train_dl, val_dl = prepare_dummy_dataloaders(
        "cifar10",
        num_crops=BASE_KWARGS["num_crops"],
        num_small_crops=0,
        num_classes=BASE_KWARGS["num_classes"],
        multicrop=False,
        batch_size=BASE_KWARGS["batch_size"],
    )
    trainer.fit(model, train_dl, val_dl)
