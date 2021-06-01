import torch
import torch.nn as nn
import torch.nn.functional as F
from solo.losses.byol import byol_loss_func
from solo.methods.base import BaseModel
from solo.utils.metrics import accuracy_at_k
from solo.utils.momentum import MomentumUpdater, initialize_momentum_params


class BYOL(BaseModel):
    def __init__(
        self,
        output_dim,
        proj_hidden_dim,
        pred_hidden_dim,
        base_tau_momentum,
        final_tau_momentum,
        **kwargs,
        ):
        super().__init__(**kwargs)

        self.last_step = 0

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_size, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, output_dim),
        )

        # predictor
        self.predictor = nn.Sequential(
            nn.Linear(output_dim, pred_hidden_dim),
            nn.BatchNorm1d(pred_hidden_dim),
            nn.ReLU(),
            nn.Linear(pred_hidden_dim, output_dim),
        )

        # instantiate and initialize momentum encoder
        self.momentum_encoder = self.base_model(
            zero_init_residual=self.zero_init_residual)
        self.momentum_encoder.fc = nn.Identity()
        if self.cifar:
            self.momentum_encoder.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=2, bias=False
            )
            self.momentum_encoder.maxpool = nn.Identity()
        initialize_momentum_params(self.encoder, self.momentum_encoder)

        # instantiate and initialize momentum projector
        self.momentum_projector = nn.Sequential(
            nn.Linear(self.features_size, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, output_dim),
        )
        initialize_momentum_params(self.projector, self.momentum_projector)

        # momentum updater
        self.momentum_updater = MomentumUpdater(base_tau_momentum, final_tau_momentum)

    @property
    def extra_learnable_params(self):
        return [{"params": self.projector.parameters()}, {"params": self.predictor.parameters()}]

    def forward(self, X):
        out = super().forward(X)
        z = self.projector(out["feat"])
        p = self.predictor(z)
        return {**out, "z": z, "p": p}

    @torch.no_grad()
    def forward_momentum(self, X):
        features_momentum = self.momentum_encoder(X)
        z_momentum = self.momentum_projector(features_momentum)
        return z_momentum

    def training_step(self, batch, batch_idx):
        indexes, (X1, X2), target = batch

        # forward online encoder
        out1 = self(X1)
        out2 = self(X2)

        z1 = out1["z"]
        z2 = out2["z"]
        p1 = out1["p"]
        p2 = out2["p"]
        logits1 = out1["logits"]
        logits2 = out2["logits"]

        # forward momentum encoder
        z1_momentum = self.forward_momentum(X1)
        z2_momentum = self.forward_momentum(X2)

        # ------- contrastive loss -------
        neg_cos_sim = byol_loss_func(p1, z2_momentum) / 2 + byol_loss_func(p2, z1_momentum) / 2

        # ------- classification loss -------
        logits = torch.cat((logits1, logits2))
        target = target.repeat(2)
        class_loss = F.cross_entropy(logits, target, ignore_index=-1)

        # just add together the losses to do only one backward()
        # we have stop gradients on the output y of the model
        loss = neg_cos_sim + class_loss

        # ------- metrics -------
        acc1, acc5 = accuracy_at_k(logits, target, top_k=(1, 5))

        z_std = F.normalize(torch.cat((z1, z2), dim=0), dim=1).std(dim=0).mean()

        metrics = {
            "train_neg_cos_sim": neg_cos_sim,
            "train_class_loss": class_loss,
            "train_z_std": z_std,
            "train_acc1": acc1,
            "train_acc5": acc5,
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        if self.trainer.global_step > self.last_step:
            # log tau momentum
            self.log("tau", self.momentum_updater.cur_tau)
            # update momentum encoder
            self.momentum_updater.update(
                online_nets=[self.encoder, self.projector],
                momentum_nets=[self.momentum_encoder, self.momentum_projector],
                cur_step=self.trainer.global_step * self.trainer.accumulate_grad_batches,
                max_steps=len(self.trainer.train_dataloader) * self.trainer.max_epochs,
            )
        self.last_step = self.trainer.global_step
