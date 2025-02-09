from functools import partial

import torch as T
import torch.nn.functional as F
from lightning import LightningModule
from torch import nn
from torchmetrics import Accuracy

from mltools.mltools.mlp import MLP
from mltools.mltools.torch_utils import ParameterNoWD
from mltools.mltools.transformers import Transformer
from src.data.utils import NUM_CSTS_ID
from src.models.utils import JetBackbone


class MaskedParticleModelling(LightningModule):
    """Class for all masked particle modelling pre-training."""

    def __init__(
        self,
        *,
        data_sample: dict,
        n_classes: int,
        embed_config: dict,
        encoder_config: dict,
        decoder_config: dict,
        optimizer: partial,
        scheduler: dict,
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

        # The transformers
        self.encoder = Transformer(**encoder_config)
        self.decoder = Transformer(**decoder_config)

        # The embedding layers
        self.cst_emb = MLP(self.cst_dim, self.encoder.dim, **embed_config)
        self.jet_emb = MLP(self.jet_dim, self.encoder.ctxt_dim, **embed_config)
        self.cst_id_emb = nn.Embedding(NUM_CSTS_ID, self.encoder.dim)

        # The linear layer to go from encoder to decoder
        self.enc_to_dec = nn.Linear(self.encoder.dim, self.decoder.dim)

        # The learnable parameters for the dropped nodes in the decoder (1 per seq)
        self.null_tokens = ParameterNoWD(T.randn((self.num_csts, self.decoder.dim)))

        # Initialise the task heads
        self.cst_id_head = nn.Linear(self.decoder.dim, NUM_CSTS_ID)
        self.cst_head = nn.Linear(self.decoder.dim, self.cst_dim)

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
        null_mask = data["null_mask"]
        jets = data["jets"]

        # Embed the inputs
        x = self.cst_emb(csts) + self.cst_id_emb(csts_id)
        ctxt = self.jet_emb(jets)

        # Pass through the encoder with the null mask
        x = self.encoder(x, mask=mask & ~null_mask, ctxt=ctxt)
        mask = self.encoder.get_combined_mask(mask)  # Might gain registers

        # Resize for the decoder
        x = self.enc_to_dec(x)

        # Trim the null tokens to seq_len and expand to match batch size
        n_reg = self.encoder.num_registers
        nt = self.null_tokens[: null_mask.size(1)]
        nt = nt.unsqueeze(0).expand(*null_mask.shape, -1)

        # Create array which allows us to index the null_mask in order per jet
        null_sorted = T.arange(null_mask.size(1), device=self.device)
        null_sorted = null_sorted.unsqueeze(0).expand_as(null_mask)
        null_sorted = null_sorted < null_mask.sum(dim=1, keepdim=True)

        # Insert the null tokens so they are ordered wrt each other
        x[:, n_reg:][null_mask] = nt[null_sorted].type(x.dtype)

        # Pass through the decoder - dont need registers after - only care about dropped
        outputs = self.decoder(x, mask=mask)[:, n_reg:][null_mask]

        # Calculate the losses using each head
        cst_loss = (self.cst_head(outputs) - csts[null_mask]).abs().mean()
        cst_id_loss = F.cross_entropy(self.cst_id_head(outputs), csts_id[null_mask])
        self.log(f"{prefix}/cst_loss", cst_loss)
        self.log(f"{prefix}/cst_id_loss", cst_id_loss)

        # Calculate the probe loss
        do_probe = batch_idx % self.probe_every == 0 or prefix == "valid"
        probe_loss = self.run_probe(data, prefix) if do_probe else 0

        # Return the total loss
        total_loss = cst_loss + cst_id_loss + probe_loss
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
        x = self.encoder(x, mask=mask, ctxt=ctxt)
        new_mask = self.encoder.get_combined_mask(mask)  # Account for registers
        return x, new_mask

    def run_probe(self, data: dict, prefix: str) -> None:
        """Run the classifier probe using a detached forward pass."""
        labels = data["labels"]
        with T.no_grad():
            x, mask = self.forward(data)
        outputs = self.probe_head(x.detach(), mask=mask)  # Detach again - to be safe ;)
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
        sched = self.hparams.scheduler(optimizer=opt, model=self)
        return [opt], [{"scheduler": sched, "interval": "step"}]

    def on_validation_epoch_end(self) -> None:
        """Create the pickled object for the backbone."""
        backbone = JetBackbone(
            cst_emb=self.cst_emb,
            cst_id_emb=self.cst_id_emb,
            jet_emb=self.jet_emb,
            encoder=self.encoder,
        )
        backbone.encoder.unpack_output = True  # Safe than sorry
        backbone.eval()
        T.save(backbone, "backbone.pkl")
