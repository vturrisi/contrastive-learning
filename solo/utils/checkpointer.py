import os
import json
from pytorch_lightning.callbacks import Callback


class Checkpointer(Callback):
    def __init__(self, args, logdir="trained_models", frequency=1):
        self.args = args
        self.logdir = logdir
        self.frequency = frequency

    def initial_setup(self, trainer):
        if trainer.logger is None:
            version = None
        else:
            version = str(trainer.logger.version)
        if version is not None:
            self.path = os.path.join(self.logdir, version)
            self.ckpt = f"{self.args.name}-{version}.ckpt"
        else:
            self.path = self.logdir
            self.ckpt = f"{self.args.name}.ckpt"

        # create logging dirs
        if trainer.is_global_zero:
            os.makedirs(self.path, exist_ok=True)

    def save_args(self, trainer):
        if trainer.is_global_zero:
            args = vars(self.args)
            json_path = os.path.join(self.path, "args.json")
            json.dump(args, open(json_path, "w"))

    def save(self, trainer):
        trainer.save_checkpoint(os.path.join(self.path, self.ckpt))

    def on_train_start(self, trainer, _):
        self.initial_setup(trainer)
        self.save_args(trainer)

    def on_validation_end(self, trainer, _):
        epoch = trainer.current_epoch
        if epoch % self.frequency == 0 and epoch != 0:
            self.save(trainer)

    def on_train_end(self, trainer, _):
        self.save(trainer)

    @staticmethod
    def add_checkpointer_args(parent_parser):
        parser = parent_parser.add_argument_group("checkpointer")
        parser.add_argument("--checkpoint_dir", default="trained_models", type=str)
        parser.add_argument("--checkpoint_frequency", default=1, type=int)
        return parent_parser
