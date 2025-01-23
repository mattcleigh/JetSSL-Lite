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
        probe_head: partial,
        probe_every: int = 50,
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
        assert self.encoder.num_registers == 1, "GPT only supports 1 register"

        # The embedding layers
        self.cst_emb = MLP(self.cst_dim, self.encoder.dim, **embed_config)
        self.jet_emb = MLP(self.jet_dim, self.encoder.ctxt_dim, **embed_config)
        self.cst_id_emb = nn.Embedding(self.vocab_size, self.encoder.dim)

        # Initialise the task heads - +1 output for the unique end-token
        self.cst_head = MLP(self.encoder.dim, self.vocab_size + 1, **embed_config)

        # Simple probe for monitoring accuracy
        self.probe_head = probe_head(inpt_dim=self.encoder.dim, outp_dim=self.n_classes)
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
        end_id = T.full((B, 1), self.vocab_size, device=csts.device, dtype=T.long)
        one_mask = T.ones((B, 1), device=mask.device, dtype=T.bool)

        # Embed the inputs (dont use tokens as inputs!)
        x = self.cst_emb(csts) + self.cst_id_emb(csts_id)
        ctxt = self.jet_emb(jets)

        # Pass through the encoder - gain 1 register for the start token
        x = self.encoder(x, mask=mask, ctxt=ctxt, causal=True)
        new_mask = self.encoder.get_combined_mask(mask)
        x = self.cst_head(x[new_mask])

        # Calculate targets
        targets = T.cat([tokens, end_id], dim=1)
        target_mask = T.cat([mask, one_mask], dim=1)
        targets = targets[target_mask]

        # Calculate the cross-entropy loss
        cst_loss = F.cross_entropy(x, targets)
        self.log(f"{prefix}/cst_loss", cst_loss)

        # Calculate the probe loss
        do_probe = batch_idx % self.probe_every == 0 or prefix == "valid"
        probe_loss = self.run_probe(data, prefix) if do_probe else 0

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

    def run_probe(self, data: dict, prefix: str) -> None:
        """Run the classifier probe using a detached forward pass."""
        labels = data["labels"]
        with T.no_grad():
            x, mask = self.forward(data)
        outputs = self.probe_head(x, mask=mask)
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
        params = filter(lambda p: p.requires_grad, self.parameters())
        opt = self.hparams.optimizer(params)
        sched = self.hparams.scheduler(optimizer=opt)
        return [opt], [{"scheduler": sched, "interval": "step"}]

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
