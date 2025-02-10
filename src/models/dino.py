import math
from copy import deepcopy
from functools import partial

import torch as T
from lightning import LightningModule
from torch import nn
from torch.nn import functional as F
from torchmetrics import Accuracy

from mltools.mltools.mlp import MLP
from mltools.mltools.torch_utils import ParameterNoWD, ema_param_sync
from mltools.mltools.transformers import Transformer
from src.data.utils import NUM_CSTS_ID
from src.models.utils import JetBackbone


def rms(x: T.Tensor, dim: int | tuple = -1, eps: float = 1e-4) -> T.Tensor:
    """Return the root mean square of the tensor, always casts to float32."""
    n = T.linalg.vector_norm(x.float(), dim=dim, keepdim=True, dtype=T.float32)
    return T.add(eps, n, alpha=math.sqrt(n.numel() / x.numel()))


def rms_norm(x, dim: int | tuple = -1, eps: float = 1e-4) -> T.Tensor:
    """Normalise the vector to have unit variance."""
    return x / rms(x, dim=dim, eps=eps).to(x.dtype)


def unit_norm(x, dim: int | tuple = -1, eps: float = 1e-4) -> T.Tensor:
    """Normalise the vector to a unit length."""
    n = T.linalg.vector_norm(x.float(), dim=dim, keepdim=True, dtype=T.float32)
    n = T.add(eps, n)
    return x / n.to(x.dtype)


def occupancy(x: T.Tensor) -> T.Tensor:
    """Calculate the occupancy / the number of unique maxima in a logit tensor."""
    return T.unique(T.argmax(x, dim=-1)).size(0) / x.shape[-1]


@T.no_grad()
def sk_center(t_out: T.Tensor, temp: float = 1, num_iters: int = 3) -> T.Tensor:
    """Apply sinkhorn-Knopp centering to ensure that sum or each row and col is 1."""
    Q = T.exp(t_out.float() / temp)  # Positive definite as these are logits
    B = Q.shape[0]  # number of samples to assign
    K = Q.shape[1]  # how many prototypes
    Q /= Q.sum()
    for _ in range(num_iters):
        Q /= Q.sum(dim=0, keepdim=True)
        Q /= K
        Q /= Q.sum(dim=1, keepdim=True)
        Q /= B
    Q *= B
    return Q


def dino2_loss(
    s_out: T.Tensor,  # student output
    t_out: T.Tensor,  # teacher output
    s_temp: float = 0.1,
    t_temp: float = 0.05,
) -> T.Tensor:
    """Calculate the loss used in the DINOv2 paper."""
    t_centered = sk_center(t_out, t_temp).detach()
    s_lsm = F.log_softmax(s_out / s_temp, dim=-1)
    loss = (t_centered * s_lsm).sum(dim=-1)
    loss = T.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)  # Badly behaved loss
    return -loss.mean()


@T.autocast("cuda", enabled=False)
def koleo_loss(x: T.Tensor, eps: float = 1e-4) -> T.Tensor:
    """Kozachenko-Leonenko entropic loss regularizer.

    From Sablayrolles et al. - 2018 - Spreading vectors for similarity search
    """
    with T.no_grad():
        dots = T.mm(x, x.t())  # Closest pair = max dot product
        dots.view(-1)[:: (x.shape[0] + 1)].fill_(-1)  # Fill the diagonal with -1
        min_idx = T.argmax(dots, dim=1)

    # Get the distance between closest pairs
    distances = F.pairwise_distance(x, x[min_idx])

    # Return the kozachenko-leonenko entropy
    loss = T.log(distances + eps)
    loss = T.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)  # Badly behaved loss
    return -loss.mean()


class MPLinear(nn.Module):
    """Magnitude Preserving Linear layer."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(unit_norm(T.randn(out_features, in_features)))

    @T.no_grad
    def force_norm(self) -> None:
        """Force normalisation of the weights."""
        self.weight.data.copy_(unit_norm(self.weight.data))

    def forward(self, x: T.Tensor) -> T.Tensor:
        return F.linear(x, unit_norm(self.weight))


class SphereMLP(nn.Module):
    """MLP for mapping between two spaces on a sphere with radius sqrt(D)."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        int_dim = (input_dim + output_dim) // 2
        self.lin1 = MPLinear(input_dim, int_dim)
        self.lin2 = MPLinear(int_dim, output_dim)
        self.lin3 = MPLinear(output_dim, output_dim)

    def forward(self, x: T.Tensor) -> T.Tensor:
        x = rms_norm(x)
        x = F.silu(self.lin1(x)) / 0.596
        x = F.silu(self.lin2(x)) / 0.596
        x = self.lin3(x)
        return unit_norm(x)  # Must be unit normed for temps to make sense


class MPLinear(nn.Module):
    """Magnitude Preserving Linear layer.

    Usefull for mapping between two spaces on a sphere.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(unit_norm(T.randn(out_features, in_features)))

    @T.no_grad
    def force_norm(self) -> None:
        """Force normalisation of the weights."""
        self.weight.data.copy_(unit_norm(self.weight.data))

    def forward(self, x: T.Tensor) -> T.Tensor:
        return F.linear(x, unit_norm(self.weight))


class JetDINO(LightningModule):
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
        ema_param_sync: float = 0.995,
        s_temp: float = 0.1,
        t_temp: float = 0.05,
        embed_dim: int = 1024,
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
        self.ema_param_sync = ema_param_sync
        self.s_temp = s_temp
        self.t_temp = t_temp
        self.embed_dim = embed_dim

        # The student model
        self.encoder = Transformer(**encoder_config)
        self.decoder = Transformer(**decoder_config)
        self.head = SphereMLP(self.encoder.dim, embed_dim)

        # The teacher model
        self.ema_encoder = deepcopy(self.encoder)
        self.ema_encoder.requires_grad_(False)
        self.ema_head = deepcopy(self.head)
        self.ema_head.requires_grad_(False)

        # The embedding layers
        self.cst_emb = MLP(self.cst_dim, self.encoder.dim, **embed_config)
        self.jet_emb = MLP(self.jet_dim, self.encoder.ctxt_dim, **embed_config)
        self.cst_id_emb = nn.Embedding(NUM_CSTS_ID, self.encoder.dim)

        # The learnable parameters for the dropped nodes in the decoder (1 per seq)
        self.null_tokens = ParameterNoWD(T.randn((self.num_csts, self.decoder.dim)))

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

        # Pass through the encoder / decoder. See mpm.py for details. This is the same.
        enc_out = self.encoder(x, mask=mask & ~null_mask, ctxt=ctxt)
        enc_mask = self.encoder.get_combined_mask(mask)
        n_reg = self.encoder.num_registers
        nt = self.null_tokens[: null_mask.size(1)]
        nt = nt.unsqueeze(0).expand(*null_mask.shape, -1)
        null_sorted = T.arange(null_mask.size(1), device=self.device)
        null_sorted = null_sorted.unsqueeze(0).expand_as(null_mask)
        null_sorted = null_sorted < null_mask.sum(dim=1, keepdim=True)
        enc_out[:, n_reg:][null_mask] = nt[null_sorted].type(enc_out.dtype)
        outputs = self.decoder(enc_out, mask=enc_mask)

        # We want the output nodes and class token - pass them through the head
        out_nodes = self.head(outputs[:, n_reg:][null_mask])
        out_cls = self.head(outputs[:, 0])

        # Pass through the ema model without any masking
        with T.no_grad():
            ema_outputs = self.ema_encoder(x, mask=mask, ctxt=ctxt)
            ema_nodes = self.ema_head(ema_outputs[:, n_reg:][null_mask])
            ema_cls = self.ema_head(ema_outputs[:, 0])

        # Calculate the loss with respect to the cls token and nodes respectively
        dino_loss = dino2_loss(out_cls, ema_cls, self.s_temp, self.t_temp)
        ibot_loss = dino2_loss(out_nodes, ema_nodes, self.s_temp, self.t_temp)
        reg_loss = koleo_loss(out_cls)
        self.log(f"{prefix}/dino_loss", dino_loss)
        self.log(f"{prefix}/ibot_loss", ibot_loss)
        self.log(f"{prefix}/reg_loss", reg_loss)

        # Log the occupancy of the teacher outputs - to test for codebook collapse
        if batch_idx % 100 == 0:
            with T.no_grad():
                cls_occ = occupancy(ema_cls)
                x_occ = occupancy(ema_nodes)
                self.log(f"{prefix}/cls_occ", cls_occ)
                self.log(f"{prefix}/x_occ", x_occ)

        # Update the ema model
        if prefix == "train":
            ema_param_sync(self.encoder, self.ema_encoder, self.ema_param_sync)
            ema_param_sync(self.head, self.ema_head, self.ema_param_sync)

        # Calculate the probe loss
        do_probe = batch_idx % self.probe_every == 0 or prefix == "valid"
        probe_loss = self.run_probe(data, prefix) if do_probe else 0

        # Return the total loss
        total_loss = dino_loss + ibot_loss + reg_loss * 0.1 + probe_loss
        self.log(f"{prefix}/total_loss", total_loss)
        return total_loss

    def forward(self, data: dict) -> T.Tensor:
        """Forward pass for inference without dropping."""
        csts = data["csts"]
        csts_id = data["csts_id"]
        mask = data["mask"]
        jets = data["jets"]

        # Inference needs the output unpacked, not always needed, but safe than sorry
        self.ema_encoder.unpack_output = True

        x = self.cst_emb(csts) + self.cst_id_emb(csts_id)
        ctxt = self.jet_emb(jets)
        x = self.ema_encoder(x, mask=mask, ctxt=ctxt)
        new_mask = self.ema_encoder.get_combined_mask(mask)  # Account for registers
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
            encoder=self.ema_encoder,
        )
        backbone.encoder.unpack_output = True  # Safe than sorry
        backbone.eval()
        T.save(backbone, "backbone.pkl")

    def normalise_weights(self) -> None:
        """Loop through all the layers and normalise the weights."""
        for m in self.modules():
            if isinstance(m, MPLinear):
                m.force_norm()

    def optimizer_step(self, *args, **kwargs) -> None:
        """Ensures that all weights are properly normalised after one step."""
        super().optimizer_step(*args, **kwargs)
        self.normalise_weights()
