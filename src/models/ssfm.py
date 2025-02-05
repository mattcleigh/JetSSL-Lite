from functools import partial

import torch as T
import torch.nn.functional as F
from lightning import LightningModule
from torch import nn
from torchmetrics import Accuracy

from mltools.mltools.mlp import MLP
from mltools.mltools.modules import Fourier
from mltools.mltools.torch_utils import append_dims
from mltools.mltools.transformers import Transformer
from src.data.utils import NUM_CSTS_ID
from src.models.utils import JetBackbone


class SetToSetFlowModelling(LightningModule):
    """Class for the set-to-set flow modelling pre-training."""

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
        self.decoder = Transformer(
            inpt_dim=self.cst_dim + NUM_CSTS_ID,  # csts and csts_id are concatenated
            outp_dim=self.cst_dim + NUM_CSTS_ID,  # Denoising, so inpt = outp
            use_decoder=True,
            **decoder_config,
        )
        self.n_reg = self.encoder.num_registers
        self.dim = self.encoder.dim

        # The embedding layers
        self.cst_emb = MLP(self.cst_dim, self.encoder.dim, **embed_config)
        self.cst_id_emb = nn.Embedding(NUM_CSTS_ID, self.encoder.dim)
        self.jet_emb = MLP(self.jet_dim, self.encoder.ctxt_dim, **embed_config)

        # The linear layer to go from encoder to decoder
        self.enc_to_dec = nn.Linear(self.encoder.dim, self.decoder.dim)

        # The decoder needs an additional embedding for the time
        self.time_emb = nn.Sequential(
            Fourier(16), MLP(16, self.decoder.ctxt_dim, **embed_config)
        )

        # Keep everything packed, this saves all the memory!!!
        # Does require and ampere GPU though...
        self.encoder.pack_inputs = True
        self.decoder.pack_inputs = True
        self.encoder.unpack_output = False
        self.decoder.unpack_output = False

        # Simple probe for monitoring accuracy
        self.probe_head = probe_head(inpt_dim=self.encoder.dim, outp_dim=self.n_classes)
        self.train_acc = Accuracy("multiclass", num_classes=n_classes)
        self.valid_acc = Accuracy("multiclass", num_classes=n_classes)

    def _shared_step(self, data: dict, batch_idx: int, prefix: str) -> T.Tensor:
        """Shared step used in both training and validaiton."""
        csts = data["csts"]
        csts_id = data["csts_id"]
        mask = data["mask"]
        null_mask = data["null_mask"]
        jets = data["jets"]

        # Training needs the output packed, this could have changed in use
        self.encoder.unpack_output = False

        # Split the jets into the two sets (done by masking only)
        enc_mask = mask & ~null_mask
        dec_mask = mask & null_mask

        # Representations for the models
        x = self.cst_emb(csts) + self.cst_id_emb(csts_id)  # Inputs
        ctxt = self.jet_emb(jets)

        # Get the output of the encoder (will be packed)
        enc_out, enc_culens, enc_maxlen = self.encoder(x, mask=enc_mask, ctxt=ctxt)

        # Resize for the decoder
        enc_out = self.enc_to_dec(enc_out)

        # Get all the values required for the diffusion / flow matching
        x0 = T.cat([csts, F.one_hot(csts_id, NUM_CSTS_ID)], dim=-1)  # Clean inputs
        t = T.sigmoid(T.randn(x0.shape[0], device=x0.device))  # Sample time
        t_emb = self.time_emb(t)  # Time embedding
        x1 = T.randn_like(x0)  # Sample noise
        t = append_dims(t, x0.ndim)  # Match dimensions for interpolation
        xt = (1 - t) * x0 + t * x1  # Interpolate
        v = x1[dec_mask] - x0[dec_mask]  # Velocity vector for target

        # Get the output of the decoder using, time and context
        v_hat, _, _ = self.decoder(
            xt,
            mask=dec_mask,
            ctxt=t_emb,
            kv=enc_out,
            kv_culens=enc_culens,
            kv_maxlen=enc_maxlen,
        )

        # Calculate the loss based on the velocity vector
        diff_loss = (v_hat - v).square().mean()
        self.log(f"{prefix}/diff_loss", diff_loss)

        # Calculate the probe loss
        do_probe = batch_idx % self.probe_every == 0 or prefix == "valid"
        probe_loss = self.run_probe(data, prefix) if do_probe else 0

        # Combine and return the losses
        total_loss = diff_loss + probe_loss
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
        sched = self.hparams.scheduler(optimizer=opt)
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
