import matplotlib.pyplot as plt
import numpy as np
import torch as T
from torch import nn

from mltools.mltools.mlp import MLP
from mltools.mltools.modules import Fourier
from mltools.mltools.torch_utils import append_dims
from mltools.mltools.transformers import Transformer


class SSFM(nn.Module):
    def __init__(
        self,
        *,
        inpt_dim: int,
        max_seq_len: int,
        embed_config: dict,
        encoder_config: dict,
        decoder_config: dict,
    ) -> None:
        super().__init__()

        # The transformers
        self.decoder = Transformer(
            inpt_dim=inpt_dim,
            outp_dim=inpt_dim * 2,  # 2 for the mean and variance
            pos_enc="abs",
            max_seq_len=max_seq_len,
            use_decoder=True,
            **decoder_config,
        )
        self.encoder = Transformer(
            inpt_dim=inpt_dim,
            outp_dim=self.decoder.dim,
            pos_enc="abs",
            max_seq_len=max_seq_len,
            **encoder_config,
        )

        # Initialise the decoder output as zeros for better convergence
        self.decoder.linear_out.weight.data.zero_()
        self.decoder.linear_out.bias.data.zero_()

        # The decoder needs an additional embedding for the time
        self.time_emb = nn.Sequential(
            Fourier(16), MLP(16, self.decoder.ctxt_dim, **embed_config)
        )

        # Make a cache for the encoder outputs for use in generation
        # Which requires a decoder loop using the same encoder outputs
        self.encoder_cache = None

    def forward(
        self,
        xt: T.Tensor,
        t: T.Tensor,
        *,  # Format here is to make it interface with the heun sampler
        x0: T.Tensor,
        xt_mask: T.BoolTensor,
        x0_mask: T.BoolTensor,
        reuse_cache: bool = False,
        return_logvar: bool = False,
    ) -> T.Tensor:
        """Get the velocity vector for the flow matching outputs."""
        if self.encoder_cache is None or not reuse_cache:
            self.encoder_cache = (
                self.encoder(x0, mask=x0_mask),
                self.encoder.get_combined_mask(x0_mask),
            )

        dec_out = self.decoder(
            xt,
            mask=xt_mask,
            ctxt=self.time_emb(t),
            kv=self.encoder_cache[0],
            kv_mask=self.encoder_cache[1],
        )
        _, v_hat = self.decoder.remove_registers(dec_out)
        v_mean, v_logvar = v_hat.chunk(2, dim=-1)
        if return_logvar:
            return v_mean, v_logvar
        v_mean[x0_mask] = 0  # Zero out the conditional velocities
        return v_mean

    def embed(self, data: T.Tensor) -> T.Tensor:
        """Embed / tokenise the input data for the encoder."""
        vals = T.stack([emb(x) for emb, x in zip(self.val_emb, data)], dim=1)
        poss = T.stack([emb(x) for emb, x in zip(self.pos_emb, data)], dim=1)
        return T.cat([vals, poss], dim=-1)

    def get_loss(self, x0: T.Tensor) -> T.Tensor:
        """Calculate the CFM loss given the clean batch."""
        B, S, _D = x0.shape

        # The data will be routed to either the encoder or decoder via the mask
        # The mask is True for encoder inputs and False for decoder inputs
        x0_mask = T.rand((B, S), device=x0.device) < 0.5
        xt_mask = ~x0_mask

        # Get all the values needed for conditional flow matching
        t = T.sigmoid(T.randn(B, device=x0.device))
        x1 = T.randn_like(x0)
        xt = x0 + (x1 - x0) * append_dims(t, x0.ndim)
        v = x1 - x0

        # Get the output from the model
        v_mean, v_logvar = self.forward(
            xt,
            t,
            x0=x0,
            xt_mask=xt_mask,
            x0_mask=x0_mask,
            return_logvar=True,
        )

        # Trim the velocity vectors to the ones that will be used in the loss
        v_mean = v_mean[xt_mask]
        v_logvar = v_logvar[xt_mask]
        v = v[xt_mask]

        # Calculate the loss based on the maximum likelihood of the velocity vector
        return ((v - v_mean).square() / v_logvar.exp() + v_logvar).mean()


@T.no_grad()
def sample_heun(
    vel_fn: callable,
    x: T.Tensor,
    times: T.Tensor,
    *args,
    save_all: bool = False,
    **kwargs,
) -> None:
    num_steps = len(times) - 1
    if save_all:
        all_stages = [x]
    time_shape = x.new_ones([x.shape[0]])
    for i in range(num_steps):
        d = vel_fn(x, times[i] * time_shape, *args, **kwargs)
        dt = times[i + 1] - times[i]
        x_2 = x + d * dt
        # Euler step if last step
        if i == num_steps - 1:
            x = x_2
        # 2nd order correction
        else:
            d_2 = vel_fn(x_2, times[i + 1] * time_shape, *args, **kwargs)
            d_prime = (d + d_2) / 2
            x = x + d_prime * dt
        if save_all:
            all_stages.append(x)
    if save_all:
        return x, all_stages
    return x


def sample_toy_data(n_samples: int, device: T.device) -> T.Tensor:
    """Samples the toy data from Appendix A.1.4."""
    theta = T.randn(n_samples, device=device) * 3
    x1 = T.randn_like(theta) * 0.5 + 2 * T.sin(theta)
    x2 = T.randn_like(theta) * 0.5 * x1.abs() + 0.1 * theta**2
    combined = T.stack([theta, x1, x2], dim=1)
    return combined.float().unsqueeze(-1)  # B x S x D = B x 3 x 1


def plot_histos(data: np.ndarray, output: np.ndarray, path: str) -> None:
    _B, S, _D = data.shape
    fig, axs = plt.subplots(1, S, figsize=(4 * S, 4))
    for i in range(S):
        data_hist, bins = np.histogram(data[:, i, 0], bins=50)
        output_hist, bins = np.histogram(output[:, i, 0], bins=bins)
        axs[i].stairs(data_hist, bins, fill=True, alpha=0.5, label="Data")
        axs[i].stairs(output_hist, bins, fill=True, alpha=0.5, label="Model")
        axs[i].set_title(f"Marginal {i + 1}")
        axs[i].legend()
    fig.savefig(path)
    plt.close()


def plot_2d_scatters(data: np.ndarray, output: np.ndarray, path: str) -> None:
    fig, axs = plt.subplots(1, 1, figsize=(4, 4))
    axs.plot(data[:, 0, 0], data[:, 1, 0], ".k", label="Data", alpha=0.1)
    axs.plot(output[:, 0, 0], output[:, 1, 0], ".r", label="Model", alpha=0.1)
    axs.set_xlim(-10, 10)
    axs.set_ylim(-5, 5)
    fig.legend()
    fig.savefig(path)
    plt.close()
