import math

import torch as T
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from mltools.mltools.transformers import Transformer


def linear_warmup_cosine_decay(
    optimizer: Optimizer,
    warmup_steps: int = 1000,
    total_steps: int = 10000,
    final_factor: float = 5e-2,
    init_factor: float = 1e-5,
) -> LambdaLR:
    """Return a scheduler with a linear warmup and a cosine decay."""
    warmup_steps = max(1, warmup_steps)  # Avoid division by zero
    assert 0 < final_factor < 1, "Final factor must be less than 1"
    assert 0 < init_factor < 1, "Initial factor must be less than 1"
    assert 0 < warmup_steps < total_steps, "Total steps must be greater than warmup"

    def fn(x: int) -> float:
        if x <= warmup_steps:
            return init_factor + x * (1 - init_factor) / warmup_steps
        if x >= total_steps:
            return final_factor
        t = (x - warmup_steps) / (total_steps - warmup_steps) * math.pi
        return (1 + math.cos(t)) * (1 - final_factor) / 2 + final_factor

    return LambdaLR(optimizer, fn)


class JetBackbone(nn.Module):
    """Generalised backbone for the jet models.

    Simply wraps the constituent embedding, constituent id embedding and encoder
    together in a single module.
    Easy for saving and loading using the pickle module.
    """

    def __init__(
        self,
        cst_emb: nn.Module,
        cst_id_emb: nn.Module,
        jet_emb: nn.Module,
        encoder: Transformer,
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.cst_emb = cst_emb
        self.cst_id_emb = cst_id_emb
        self.jet_emb = jet_emb
        self.encoder = encoder
        self.causal = causal
        self.dim = encoder.dim
        self.outp_dim = encoder.outp_dim

    def forward(
        self,
        csts: T.Tensor,
        csts_id: T.Tensor,
        mask: T.Tensor,
        jets: T.Tensor,
    ) -> T.Tensor:
        """Pass through the complete backbone."""
        csts = self.cst_emb(csts) + self.cst_id_emb(csts_id)
        jets = self.jet_emb(jets)
        x = self.encoder(csts, mask=mask, ctxt=jets, causal=self.causal)
        new_mask = self.encoder.get_combined_mask(mask)  # Registers
        return x, new_mask
