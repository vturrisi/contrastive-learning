# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import inspect
import os

import hydra
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.strategies.ddp import DDPStrategy
from omegaconf import DictConfig, OmegaConf
from solo.args.pretrain import parse_cfg
from solo.data.classification_dataloader import prepare_data as prepare_data_classification
from solo.data.pretrain_dataloader import (
    FullTransformPipeline,
    NCropAugmentation,
    build_transform_pipeline,
    prepare_dataloader,
    prepare_datasets,
)
from solo.methods import METHODS
from solo.utils.auto_resumer import AutoResumer
from solo.utils.checkpointer import Checkpointer
from solo.utils.misc import make_contiguous, omegaconf_select

try:
    from solo.data.dali_dataloader import PretrainDALIDataModule, build_transform_pipeline_dali
except ImportError:
    _dali_avaliable = False
else:
    _dali_avaliable = True

try:
    from solo.utils.auto_umap import AutoUMAP
except ImportError:
    _umap_available = False
else:
    _umap_available = True


@hydra.main(version_base="1.2")
def main(cfg: DictConfig):
    # hydra doesn't allow us to add new keys for "safety"
    # set_struct(..., False) disables this behavior and allows us to add more parameters
    # without making the user specify every single thing about the model
    OmegaConf.set_struct(cfg, False)
    cfg = parse_cfg(cfg)

    seed_everything(cfg.seed)

    assert cfg.method in METHODS, f"Choose from {METHODS.keys()}"

    if cfg.data.num_large_crops != 2:
        assert cfg.method in ["wmse", "mae"]

    model = METHODS[cfg.method](cfg)
    make_contiguous(model)
    # can provide up to ~20% speed up
    if not cfg.performance.disable_channel_last:
        model = model.to(memory_format=torch.channels_last)

    # validation dataloader for when it is available
    if cfg.data.dataset == "custom" and (cfg.data.no_labels or cfg.data.val_path is None):
        val_loader = None
    elif cfg.data.dataset in ["imagenet100", "imagenet"] and cfg.data.val_path is None:
        val_loader = None
    else:
        if cfg.data.format == "dali":
            val_data_format = "image_folder"
        else:
            val_data_format = cfg.data.format

        _, val_loader = prepare_data_classification(
            cfg.data.dataset,
            train_data_path=cfg.data.train_path,
            val_data_path=cfg.data.val_path,
            data_format=val_data_format,
            batch_size=cfg.optimizer.batch_size,
            num_workers=cfg.data.num_workers,
        )

    # pretrain dataloader
    if cfg.data.format == "dali":
        assert (
            _dali_avaliable
        ), "Dali is not currently avaiable, please install it first with pip3 install .[dali]."
        pipelines = []
        for aug_cfg in cfg.augmentations:
            pipelines.append(
                NCropAugmentation(
                    build_transform_pipeline_dali(
                        cfg.data.dataset, aug_cfg, dali_device=cfg.dali.device
                    ),
                    aug_cfg.num_crops,
                )
            )
        transform = FullTransformPipeline(pipelines)

        dali_datamodule = PretrainDALIDataModule(
            dataset=cfg.data.dataset,
            train_data_path=cfg.data.train_path,
            transforms=transform,
            num_large_crops=cfg.data.num_large_crops,
            num_small_crops=cfg.data.num_small_crops,
            num_workers=cfg.data.num_workers,
            batch_size=cfg.optimizer.batch_size,
            no_labels=cfg.data.no_labels,
            data_fraction=cfg.data.fraction,
            dali_device=cfg.dali.device,
            encode_indexes_into_labels=cfg.dali.encode_indexes_into_labels,
        )
        dali_datamodule.val_dataloader = lambda: val_loader
    else:
        pipelines = []
        for aug_cfg in cfg.augmentations:
            pipelines.append(
                NCropAugmentation(
                    build_transform_pipeline(cfg.data.dataset, aug_cfg), aug_cfg.num_crops
                )
            )
        transform = FullTransformPipeline(pipelines)

        if cfg.debug_augmentations:
            print("Transforms:")
            print(transform)

        train_dataset = prepare_datasets(
            cfg.data.dataset,
            transform,
            train_data_path=cfg.data.train_path,
            data_format=cfg.data.format,
            no_labels=cfg.data.no_labels,
            data_fraction=cfg.data.fraction,
        )
        train_loader = prepare_dataloader(
            train_dataset, batch_size=cfg.optimizer.batch_size, num_workers=cfg.data.num_workers
        )

    # 1.7 will deprecate resume_from_checkpoint, but for the moment
    # the argument is the same, but we need to pass it as ckpt_path to trainer.fit
    ckpt_path, wandb_run_id = None, None
    if cfg.auto_resume.enabled and cfg.resume_from_checkpoint is None:
        auto_resumer = AutoResumer(
            checkpoint_dir=os.path.join(cfg.checkpoint.dir, cfg.method),
            max_hours=cfg.auto_resume.max_hours,
        )
        resume_from_checkpoint, wandb_run_id = auto_resumer.find_checkpoint(cfg)
        if resume_from_checkpoint is not None:
            print(
                "Resuming from previous checkpoint that matches specifications:",
                f"'{resume_from_checkpoint}'",
            )
            ckpt_path = resume_from_checkpoint
    elif cfg.resume_from_checkpoint is not None:
        ckpt_path = cfg.resume_from_checkpoint
        del cfg.resume_from_checkpoint

    callbacks = []

    if cfg.checkpoint.enabled:
        ckpt = Checkpointer(
            cfg,
            logdir=os.path.join(cfg.checkpoint.dir, cfg.method),
            frequency=cfg.checkpoint.frequency,
            keep_prev=cfg.checkpoint.keep_prev,
        )
        callbacks.append(ckpt)

    if omegaconf_select(cfg, "auto_umap.enabled", False):
        assert (
            _umap_available
        ), "UMAP is not currently avaiable, please install it first with [umap]."
        auto_umap = AutoUMAP(
            cfg.name,
            logdir=os.path.join(cfg.auto_umap.dir, cfg.method),
            frequency=cfg.auto_umap.frequency,
        )
        callbacks.append(auto_umap)

    # wandb logging
    if cfg.wandb.enabled:
        wandb_logger = WandbLogger(
            name=cfg.name,
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            offline=cfg.wandb.offline,
            resume="allow" if wandb_run_id else None,
            id=wandb_run_id,
        )
        wandb_logger.watch(model, log="gradients", log_freq=100)
        wandb_logger.log_hyperparams(OmegaConf.to_container(cfg))

        # lr logging
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)

    trainer_kwargs = OmegaConf.to_container(cfg)
    # we only want to pass in valid Trainer args, the rest may be user specific
    valid_kwargs = inspect.signature(Trainer.__init__).parameters
    trainer_kwargs = {name: trainer_kwargs[name] for name in valid_kwargs if name in trainer_kwargs}
    trainer_kwargs.update(
        {
            "logger": wandb_logger if cfg.wandb.enabled else None,
            "callbacks": callbacks,
            "enable_checkpointing": False,
            "strategy": (
                DDPStrategy(find_unused_parameters=False) if cfg.strategy == "ddp" else cfg.strategy
            ),
        }
    )
    trainer = Trainer(**trainer_kwargs)

    if cfg.data.format == "dali":
        trainer.fit(model, ckpt_path=ckpt_path, datamodule=dali_datamodule)
    else:
        trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
