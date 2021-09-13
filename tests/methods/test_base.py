import pytest
import torch
from solo.methods.base import BaseMethod

from .utils import DATA_KWARGS, gen_base_kwargs


def test_base():
    BASE_KWARGS = gen_base_kwargs(cifar=False)
    kwargs = {
        **BASE_KWARGS,
        **DATA_KWARGS,
    }
    model = BaseMethod(**kwargs)

    # test optimizers/scheduler
    model.optimizer = "random"
    model.scheduler = "none"
    with pytest.raises(ValueError):
        model.configure_optimizers()

    model.optimizer = "sgd"
    model.scheduler = "none"
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Optimizer)

    model.scheduler = "cosine"
    scheduler = model.configure_optimizers()[1][0]
    assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)

    model.scheduler = "step"
    scheduler = model.configure_optimizers()[1][0]
    assert isinstance(scheduler, torch.optim.lr_scheduler.MultiStepLR)

    model.scheduler = "random"
    with pytest.raises(ValueError):
        model.configure_optimizers()

    model.lars = False

    model.optimizer = "adam"
    model.scheduler = "none"
    model.extra_optimizer_args = {}
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Optimizer)

    model.optimizer = "adamw"
    model.scheduler = "none"
    model.extra_optimizer_args = {}
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Optimizer)
