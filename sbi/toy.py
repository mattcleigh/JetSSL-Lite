"""A quick playground for testing Simulation Based Inference.

https://arxiv.org/pdf/2404.09636

"""

import argparse
import math
from copy import deepcopy

import rootutils
import torch as T
from torch.optim.lr_scheduler import LambdaLR
from tqdm import trange

root = rootutils.setup_root(search_from=__file__, pythonpath=True)

from mltools.mltools.optimisers import AdamWS
from mltools.mltools.torch_utils import count_parameters, ema_param_sync
from sbi.utils import SSFM, plot_2d_scatters, plot_histos, sample_heun, sample_toy_data


def import_args():
    parser = argparse.ArgumentParser(
        description="Toy model arguments",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10_000,
        help="Batch size",
    )
    parser.add_argument(
        "--total_steps",
        type=int,
        default=20000,
        help="The number of training steps to run for",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Maximum learning rate (after warmup)",
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
        help="Gradient clipping",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.99,
        help="The exponential moving average decay for the offline diffusion model",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=1000,
        help="The number of learning rate warmup steps",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Which device to use (cuda or cpu)",
    )
    parser.add_argument(
        "--plot_interval",
        type=int,
        default=1000,
        help="How often to plot",
    )
    parser.add_argument(
        "--inference_samples",
        type=int,
        default=5000,
        help="The number of samples for inference and plotting",
    )
    parser.add_argument(
        "--integration_steps",
        type=int,
        default=200,
        help="The number of integration steps for the Heun sampler",
    )

    return parser.parse_args()


def main():
    args = import_args()

    # Set the device
    device = T.device(
        "cuda" if T.cuda.is_available() and args.device == "cuda" else "cpu"
    )
    integration_times = T.linspace(1, 0, args.integration_steps)

    # Initialise the model
    model = SSFM(
        inpt_dim=1,
        max_seq_len=3,
        embed_config={"num_blocks": 2, "hddn_dim": 64},
        encoder_config={"num_layers": 3, "dim": 64, "num_registers": 1},
        decoder_config={"num_layers": 3, "dim": 64, "num_registers": 1, "ctxt_dim": 16},
    )
    model.to(device)
    ema_model = deepcopy(model)
    ema_model.requires_grad_(False)
    print(f"Model has {count_parameters(model)} parameters.")

    # Initialise the optimiser and scheduler
    def fn(x: int) -> float:
        if x <= args.warmup_steps:
            return x / args.warmup_steps
        if x >= args.total_steps:
            return 0.0
        t = (x - args.warmup_steps) / (args.total_steps - args.warmup_steps) * math.pi
        return (1 + math.cos(t)) / 2

    optim = AdamWS(model.parameters(), lr=args.lr)
    sched = LambdaLR(optim, fn)

    # Training
    pbar = trange(args.total_steps)
    for it in pbar:
        model.train()
        optim.zero_grad()

        # Get training samples
        x = sample_toy_data(args.batch_size, device)

        # Get loss
        loss = model.get_loss(x)

        # Gradient step
        loss.backward()
        T.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optim.step()
        sched.step()

        # Update the EMA model
        ema_param_sync(model, ema_model, args.ema_decay)

        # Set the tqdm bar to show the loss and the lr
        lr = optim.param_groups[0]["lr"]
        pbar.set_postfix(loss=loss.item(), lr=lr, refresh=False)

        # Plot some intermediate results
        if it % args.plot_interval == 0:
            ema_model.eval()

            # Do the fully marginal case - everything goes to decoder (x0_mask = False)
            x = sample_toy_data(args.inference_samples, device)
            x0_mask = T.zeros((args.inference_samples, 3), dtype=T.bool, device=device)
            x_model = T.randn_like(x)  # Not allowed to see anything, lets make sure!

            ema_model.encoder_cache = None
            output = sample_heun(
                ema_model,
                x_model,
                integration_times,
                x0=x_model,
                x0_mask=x0_mask,
                xt_mask=~x0_mask,
                reuse_cache=True,
            )
            x = x.cpu().detach().numpy()
            output = output.cpu().detach().numpy()
            plot_histos(x, output, root / "sbi/uncond_histos.png")
            plot_2d_scatters(x, output, root / "sbi/uncond_scatter.png")

            # Do the conditional case - p(x2, theta | x1 in [2, 3])
            # The mask is [False, True, False] for [theta, x1, x2] respectively
            x = sample_toy_data(args.inference_samples * 5, device)  # Oversample!
            x = x[x[:, 1, 0] > 2]  # Rejection sample!
            x = x[x[:, 1, 0] < 3]
            x0_mask = T.tensor([0, 1, 0], dtype=bool, device=device)
            x0_mask = x0_mask.expand(x.shape[0], 3)
            x_model = T.randn_like(x)
            x_model[x0_mask] = x[x0_mask]

            ema_model.encoder_cache = None
            output = sample_heun(
                ema_model,
                x_model,
                integration_times,
                x0=x_model,
                x0_mask=x0_mask,
                xt_mask=~x0_mask,
                reuse_cache=True,
            )
            x = x.cpu().detach().numpy()
            output = output.cpu().detach().numpy()
            plot_histos(x, output, root / "sbi/cond_histos.png")
            plot_2d_scatters(x, output, root / "sbi/cond_scatter.png")


if __name__ == "__main__":
    main()
