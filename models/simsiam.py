import os
import sys

# import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from base import Model
except:
    from .base import Model

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from losses.neg_cosine_sim import negative_cosine_similarity
from utils.gather_layer import gather
from utils.metrics import accuracy_at_k


class SimSiam(Model):
    def __init__(self, args):
        super().__init__(args)

        proj_hidden_mlp = args.hidden_mlp
        output_dim = args.encoding_size

        pred_hidden_mlp = args.pred_hidden_mlp

        # projection head
        self.projection_head = nn.Sequential(
            nn.Linear(self.encoder.n_features, proj_hidden_mlp),
            nn.BatchNorm1d(proj_hidden_mlp),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden_mlp, proj_hidden_mlp),
            nn.BatchNorm1d(proj_hidden_mlp),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden_mlp, output_dim),
            nn.BatchNorm1d(output_dim),
        )

        # prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(output_dim, pred_hidden_mlp),
            nn.BatchNorm1d(pred_hidden_mlp),
            nn.ReLU(inplace=True),
            nn.Linear(pred_hidden_mlp, output_dim),
        )

    def forward(self, X, classify_only=True):
        features, y = super().forward(X, classify_only=False)
        if classify_only:
            return y
        else:
            z = self.projection_head(features)
            p = self.prediction_head(z)
            return features, z, p, y

    def training_step(self, batch, batch_idx):
        indexes, (X_aug1, X_aug2), target = batch

        # features, projection head features, class
        features, z1, p1, output = self(X_aug1, classify_only=False)
        features, z2, p2, output = self(X_aug2, classify_only=False)

        z1 = gather(z1)
        z2 = gather(z2)

        p1 = gather(p1)
        p2 = gather(p2)

        # ------- contrastive loss -------
        neg_cos_sim = (
            negative_cosine_similarity(p1, z2) / 2 + negative_cosine_similarity(p2, z1) / 2
        )

        # ------- classification loss -------
        # for datasets with unsupervised data
        index = target >= 0
        output = output[index]
        target = target[index]

        # ------- classification loss -------
        class_loss = F.cross_entropy(output, target)

        # just add together the losses to do only one backward()
        # we have stop gradients on the output y of the model
        loss = neg_cos_sim + class_loss

        # ------- metrics -------
        acc1, acc5 = accuracy_at_k(output, target, top_k=(1, 5))

        metrics = {
            "train_neg_cos_sim": neg_cos_sim,
            "train_class_loss": class_loss,
            "train_acc1": acc1,
            "train_acc5": acc5,
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)
        return loss
