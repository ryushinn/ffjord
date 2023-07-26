import argparse
import os
import time
from datetime import datetime
import random
import numpy as np

import os.path as opath
import copy
import torch
import torch.optim as optim
import torchvision.io as tvio
import torchvision.datasets as dset
import torchvision.transforms as tforms
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
from tqdm import tqdm

import torch.nn as nn
import net
import metrics
import lib.layers as layers
import lib.utils as utils
import lib.odenvp as odenvp
import lib.multiscale_parallel as multiscale_parallel

parser = argparse.ArgumentParser("ODE texture")

SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", "adams", "explicit_adams"]
parser.add_argument("--solver", type=str, default="dopri5", choices=SOLVERS)
parser.add_argument("--atol", type=float, default=1e-5)
parser.add_argument("--rtol", type=float, default=1e-5)

_converter = lambda s: tuple(map(int, s.split(",")))
parser.add_argument("--dims", type=_converter, default="64,128,256")
parser.add_argument("--LDM", action="store_true")
parser.add_argument("--num_layers", type=int, default=3, required=False)
parser.add_argument("--loss_type", type=str, choices=["GRAM", "SW"], default="GRAM")

parser.add_argument("--eps", type=float, default=1e-6)
parser.add_argument("--num_epochs", type=int, default=1000)
parser.add_argument("--num_disp_epochs", type=int, default=10)
parser.add_argument("--batchsize", type=int, default=1)
parser.add_argument("--lr", type=float, default=5e-4)

parser.add_argument("--exemplar_path", type=str)
parser.add_argument("--exp_path", type=str)
parser.add_argument("--comment", type=str, default="")

# args
args = parser.parse_args()


def seed_all(seed):
    """
    provide the seed for reproducibility
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == "__main__":
    # pre, reproducible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cvt = lambda x: x.to(device)
    torch.set_default_dtype(torch.float32)

    seed_all(42)

    # data
    if not opath.exists(args.exemplar_path):
        raise ValueError(f"There is not file in {args.exemplar_path}")

    ## create workspace to store results
    workspace = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    ws_path = opath.join(args.exp_path, workspace) + args.comment
    writer = SummaryWriter(log_dir=ws_path)

    read_tforms = [
        # tforms.RandomCrop(128),
        tforms.ConvertImageDtype(torch.float32),
    ]

    exemplar = tvio.read_image(args.exemplar_path)
    for tform in read_tforms:
        exemplar = tform(exemplar)

    ## write transformed exemplar
    writer.add_image("exemplar.png", exemplar)

    exemplar = cvt(exemplar)
    data_shape = exemplar.size()
    exemplar = torch.unsqueeze(exemplar, 0)  # add the batch dim

    # model
    if args.LDM:
        odefunc = net.UNet(inout_channels=3 * (args.num_layers + 1), channels=args.dims)
        model = nn.Sequential(
            net.Lambda(net.compile),
            net.ODETexture(
                odefunc, T=1, solver=args.solver, atol=args.atol, rtol=args.rtol
            ),
            net.Lambda(net.decompile),
            net.Lambda(net.decoder),
            net.SigmoidTransform(args.eps),
        )
    else:
        odefunc = net.UNet(inout_channels=3, channels=args.dims)
        model = nn.Sequential(
            net.ODETexture(
                odefunc, T=1, solver=args.solver, atol=args.atol, rtol=args.rtol
            ),
            net.SigmoidTransform(args.eps),
        )

    model = cvt(model)

    # training preconfig

    ## VGG features
    features = metrics.VGGFeatures().to(device)
    features_exemplar = features(exemplar)
    gmatrices_exemplar = list(map(metrics.GramMatrix, features_exemplar))
    loss_fn = nn.MSELoss(reduction="mean")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    n_test_tex = 3
    test_noise_size = (512, 512)
    if args.LDM:
        test_noise = net.rand_pyramid(
            [n_test_tex, 3, *test_noise_size], args.num_layers, device
        )
    else:
        test_noise = torch.randn(n_test_tex, 3, *test_noise_size, device=device)

    noise_size = (256, 256)
    # training procedure
    with tqdm(total=args.num_epochs, desc="Epoch") as t:
        for ep in range(args.num_epochs):
            model.train()
            optimizer.zero_grad()

            ## generate some textures
            if args.LDM:
                noise = net.rand_pyramid(
                    [args.batchsize, 3, *noise_size], args.num_layers, device
                )
            else:
                noise = torch.randn(args.batchsize, 3, *noise_size, device=device)
            generated_textures = model(noise)

            ## compute gram matrices
            features_samples = features(generated_textures)
            gmatrices_samples = list(map(metrics.GramMatrix, features_samples))

            ## compute the gradients
            if args.loss_type == "GRAM":
                loss = 0.0
                for gmatrix_e, gmatrix_s in zip(gmatrices_exemplar, gmatrices_samples):
                    loss += loss_fn(gmatrix_e.expand_as(gmatrix_s), gmatrix_s)
            elif args.loss_type == "SW":
                loss = metrics.SlicedWassersteinLoss(
                    features_exemplar, features_samples
                )

            loss.backward()

            optimizer.step()

            writer.add_scalar("training_loss", loss.item(), ep)

            ## test in some period
            if ep % args.num_disp_epochs == 0:
                with torch.no_grad():
                    model.eval()
                    tex = model(test_noise).to("cpu")
                    writer.add_image("test_tex", make_grid(tex, nrow=3), ep)

                torch.save(
                    copy.deepcopy(model.state_dict()),
                    opath.join(ws_path, f"model_checkpoint.pth"),
                )

            t.update()
