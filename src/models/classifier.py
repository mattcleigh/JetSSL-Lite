import logging
from functools import partial
from typing import TYPE_CHECKING

import torch as T
from lightning import LightningModule
from torch.nn.functional import cross_entropy
from torchmetrics import Accuracy

if TYPE_CHECKING:
    from src.models.utils import JetBackbone

log = logging.getLogger(__name__)

# TODO(Matthew): Make this a parameter... somehow
# 001
CSTS_ID = 8


class Classifier(LightningModule):
    """A class for fine tuning a classifier based on a model with an encoder.

    This should be paired with a scheduler for unfreezing/freezing the backbone.
    """

    def __init__(
        self,
        *,
        data_sample: tuple,
        n_classes: int,
        backbone_path: str,
        class_head: partial,
        optimizer: partial,
        scheduler: partial,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.n_classes = n_classes

        # Load the pretrained and pickled JetBackbone object.
        log.info(f"Loading backbone from {backbone_path}")
        self.backbone: JetBackbone = T.load(backbone_path, map_location="cpu")

        # Create the head for the downstream task
        self.class_head = class_head(
            inpt_dim=self.backbone.encoder.outp_dim,
            outp_dim=n_classes,
        )

        # Metrics
        self.train_acc = Accuracy("multiclass", num_classes=n_classes)
        self.valid_acc = Accuracy("multiclass", num_classes=n_classes)

    def forward(
        self,
        csts: T.Tensor,
        csts_id: T.Tensor,
        mask: T.BoolTensor,
        jets: T.Tensor,
    ) -> T.Tensor:
        x, mask = self.backbone(csts, csts_id, mask, jets)  # Might gain registers
        return self.class_head(x, mask=mask)

    def _shared_step(self, data: tuple, prefix: str) -> T.Tensor:
        """Shared step used in both training and validaiton."""
        csts = data["csts"]
        csts_id = data["csts_id"]
        labels = data["labels"]
        mask = data["mask"]
        jets = data["jets"]

        output = self.forward(csts, csts_id, mask, jets)
        loss = cross_entropy(output, labels, label_smoothing=0.1)
        self.log(f"{prefix}/total_loss", loss)

        acc = getattr(self, f"{prefix}_acc")
        acc(output, labels)
        self.log(f"{prefix}/acc", acc)

        return loss

    def training_step(self, data: dict) -> T.Tensor:
        return self._shared_step(data, "train")

    def validation_step(self, data: dict) -> T.Tensor:
        return self._shared_step(data, "valid")

    def predict_step(self, data: dict) -> T.Tensor:
        """Return a dictionary for saving final exported scores."""
        csts = data["csts"]
        csts_id = data["csts_id"]
        labels = data["labels"]
        mask = data["mask"]
        jets = data["jets"]
        output = self.forward(csts, csts_id, mask, jets)
        return {"output": output, "label": labels.unsqueeze(-1)}

    def configure_optimizers(self) -> dict:
        params = filter(lambda p: p.requires_grad, self.parameters())
        opt = self.hparams.optimizer(params)
        sched = self.hparams.scheduler(optimizer=opt, model=self)
        return [opt], [{"scheduler": sched, "interval": "step"}]
