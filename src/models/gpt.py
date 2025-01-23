from functools import partial

import torch as T
import torch.nn.functional as F
from lightning import LightningModule
from torch import nn
from torchmetrics import Accuracy

from mltools.mltools.mlp import MLP
from mltools.mltools.transformers import Transformer
from src.data.utils import NUM_CSTS_ID
from src.models.utils import JetBackbone


class JetGPT(LightningModule):
    """Class for generative pre-training of transformers for jet data."""

    def __init__(
        self,
        *,
        data_sample: dict,
        n_classes: int,
        embed_config: dict,
        encoder_config: dict,
        optimizer: partial,
        scheduler: dict,
        vocab_size: int,
        probe_every: int = 100,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        # Break down the data sample into the dimensions needed for the model
        self.cst_dim = data_sample["csts"].shape[-1]
        self.jet_dim = data_sample["jets"].shape[-1]
        self.num_csts = data_sample["csts"].shape[-2]

        # Attributes
        self.n_classes = n_classes
        self.probe_every = probe_every
        self.vocab_size = vocab_size * NUM_CSTS_ID  # Combined token space

        # The transformers
        self.encoder = Transformer(**encoder_config)

        # The embedding layers
        self.cst_emb = MLP(self.cst_dim, self.encoder.dim, **embed_config)
        self.jet_emb = MLP(self.jet_dim, self.encoder.ctxt_dim, **embed_config)
        self.cst_id_emb = nn.Embedding(self.vocab_size, self.encoder.dim)

        # Initialise the task heads - +1 output for the unique end-token
        self.cst_head = MLP(self.encoder.dim, self.vocab_size + 1, **embed_config)

        # The start token
        self.start_token = nn.Parameter(T.randn(self.encoder.dim))

        # Simple linear probe for monitoring accuracy
        self.probe = nn.Linear(self.encoder.outp_dim, self.n_classes)
        self.train_acc = Accuracy("multiclass", num_classes=n_classes)
        self.valid_acc = Accuracy("multiclass", num_classes=n_classes)

        self.on_validation_epoch_end()  # Test saving the backbone

    def _shared_step(self, data: dict, batch_idx: int, prefix: str) -> T.Tensor:
        """Shared step used in both training and validaiton."""
        csts = data["csts"]
        csts_id = data["csts_id"]
        mask = data["mask"]
        jets = data["jets"]
        tokens = data["tokens"]
        B = mask.shape[0]

        # Create the needed elements for the autoregressive model
        start_token = self.start_token.expand(B, 1, -1)
        end_id = T.full((B, 1), self.vocab_size, device=csts.device, dtype=T.long)
        one_mask = T.ones((B, 1), device=mask.device, dtype=T.bool)

        # Embed the inputs (dont use tokens as inputs!)
        x = self.cst_emb(csts) + self.cst_id_emb(csts_id)
        ctxt = self.jet_emb(jets)

        # Add the start token to the beginning of the sequence
        x = T.cat([start_token, x], dim=1)
        input_mask = T.cat([one_mask, mask], dim=1)

        # Pass through the encoder
        x = self.encoder(x, mask=input_mask, ctxt=ctxt, causal=True)
        x = self.cst_head(x[input_mask])

        # Calculate targets
        targets = T.cat([tokens, end_id], dim=1)
        target_mask = T.cat([mask, one_mask], dim=1)
        targets = targets[target_mask]

        # Calculate the cross-entropy loss
        cst_loss = F.cross_entropy(x, targets)
        self.log(f"{prefix}/cst_loss", cst_loss)

        # Calculate the probe loss
        do_probe = batch_idx % self.probe_every == 0 or prefix == "valid"
        probe_loss = self.linear_probe(data, prefix) if do_probe else 0

        # Return the total loss
        total_loss = cst_loss + probe_loss
        self.log(f"{prefix}/total_loss", total_loss)
        return total_loss

    def forward(self, data: dict) -> T.Tensor:
        """Forward pass for inference without dropping."""
        csts = data["csts"]
        csts_id = data["csts_id"]
        mask = data["mask"]
        jets = data["jets"]

        # Inference needs the output unpacked, not always needed, but safe than sorry
        self.encoder.unpack_output = True

        x = self.cst_emb(csts) + self.cst_id_emb(csts_id)
        ctxt = self.jet_emb(jets)
        x = self.encoder(x, mask=mask, ctxt=ctxt, causal=True)
        new_mask = self.encoder.get_combined_mask(mask)  # Account for registers
        return x, new_mask

    def linear_probe(self, data: dict, prefix: str) -> None:
        """Do the linear probe using a detached forward pass."""
        labels = data["labels"]
        with T.no_grad():
            outputs, outmask = self.forward(data)
            outputs = (outputs * outmask.unsqueeze(-1)).mean(-2)
        outputs = self.probe(outputs)
        probe_loss = F.cross_entropy(outputs, labels)
        acc = getattr(self, f"{prefix}_acc")
        acc(outputs, labels)
        self.log(f"{prefix}/probe_accuracy", acc)
        self.log(f"{prefix}/probe_loss", probe_loss)
        return probe_loss

    def training_step(self, data: dict, batch_idx: int) -> T.Tensor:
        return self._shared_step(data, batch_idx, "train")

    def validation_step(self, data: dict, batch_idx: int) -> T.Tensor:
        return self._shared_step(data, batch_idx, "valid")

    def configure_optimizers(self) -> dict:
        opt = self.hparams.optimizer(
            filter(lambda p: p[1].requires_grad, self.named_parameters())
        )
        scheduler = {
            "scheduler": self.hparams.scheduler(optimizer=opt),
            "interval": "step",
        }
        return [opt], [scheduler]

    def on_validation_epoch_end(self) -> None:
        """Create the pickled object for the backbone."""
        backbone = JetBackbone(
            cst_emb=self.cst_emb,
            cst_id_emb=self.cst_id_emb,
            jet_emb=self.jet_emb,
            encoder=self.encoder,
            causal=True,
        )
        backbone.encoder.unpack_output = True  # Safe than sorry
        backbone.eval()
        T.save(backbone, "backbone.pkl")
