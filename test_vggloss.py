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
from torchvision.utils import save_image, make_grid
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

parser = argparse.ArgumentParser("ODE texture direct")

parser.add_argument("--exemplar_path", type=str)
parser.add_argument("--exp_path", type=str)
parser.add_argument("--lr", type=float, default=1e-4)

parser.add_argument("--num_epochs", type=int, default=50000)
parser.add_argument("--num_disp_epochs", type=int, default=500)
parser.add_argument("--comment", type=str, default="")
parser.add_argument("--loss_type", type=str, choices=["GRAM", "SW"], default="GRAM")

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

    noise = torch.logit(exemplar.mean((2, 3), keepdim=True) + 0.01 * torch.randn(1, 3, 512, 512).to(device))
    noise.requires_grad_(True)
    # training preconfig

    ## VGG features
    features = metrics.VGGFeatures().to(device)
    features_exemplar = features(exemplar)
    gmatrices_exemplar = list(map(metrics.GramMatrix, features_exemplar))
    loss_fn = nn.MSELoss(reduction="mean")

    optimizer = optim.Adam([noise], lr=args.lr)

    # training procedure
    with tqdm(total=args.num_epochs, desc="Epoch") as t:
        for ep in range(args.num_epochs):
            optimizer.zero_grad()

            ## compute gram matrices
            features_noise = features(torch.sigmoid(noise))
            gmatrices_samples = list(map(metrics.GramMatrix, features_noise))

            ## compute the gradients
            if args.loss_type == 'GRAM':
                loss = 0.0
                for gmatrix_e, gmatrix_s in zip(gmatrices_exemplar, gmatrices_samples):
                    loss += loss_fn(gmatrix_e.expand_as(gmatrix_s), gmatrix_s)
            elif args.loss_type == 'SW':
                loss = metrics.SlicedWassersteinLoss(features_exemplar, features_noise)

            loss.backward()

            optimizer.step()

            writer.add_scalar("training_loss", loss.item(), ep)

            if ep % args.num_disp_epochs == 0:
                with torch.no_grad():
                    writer.add_image(
                        "optimized_noise", make_grid(torch.sigmoid(noise), nrow=3), ep
                    )

            t.update()
            ## test in some period
