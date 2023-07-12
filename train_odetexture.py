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
from torchvision.utils import save_image

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
parser.add_argument("--dims", type=_converter, default="8,32,32,8")
parser.add_argument("--strides", type=_converter, default="2,2,1,-2,-2")
parser.add_argument("--num_blocks", type=int, default=1, help="Number of stacked CNFs.")

parser.add_argument("--conv", type=eval, default=True, choices=[True, False])
parser.add_argument(
    "--layer_type",
    type=str,
    default="ignore",
    choices=[
        "ignore",
        "concat",
        "concat_v2",
        "squash",
        "concatsquash",
        "concatcoord",
        "hyper",
        "blend",
    ],
)
parser.add_argument(
    "--nonlinearity",
    type=str,
    default="softplus",
    choices=["tanh", "relu", "softplus", "elu", "swish"],
)

parser.add_argument("--alpha", type=float, default=1e-6)
parser.add_argument("--num_epochs", type=int, default=1000)
parser.add_argument("--num_disp_epochs", type=int, default=10)
parser.add_argument("--batchsize", type=int, default=10)
parser.add_argument("--lr", type=float, default=1e-4)

parser.add_argument("--exemplar_path", type=str)
parser.add_argument("--exp_path", type=str)

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
    ws_path = opath.join(args.exp_path, workspace)
    os.makedirs(ws_path, exist_ok=True)

    read_tforms = [
        # tforms.RandomCrop(128),
        tforms.ConvertImageDtype(torch.float32),
    ]

    exemplar = tvio.read_image(args.exemplar_path)
    for tform in read_tforms:
        exemplar = tform(exemplar)

    ## write transformed exemplar
    tvio.write_png(
        tforms.ConvertImageDtype(torch.uint8)(exemplar),
        opath.join(ws_path, "exemplar.png"),
    )

    exemplar = cvt(exemplar)
    data_shape = exemplar.size()
    exemplar = torch.unsqueeze(exemplar, 0)  # add the batch dim

    # model

    # model = odenvp.ODENVP(
    #     (args.batchsize, *data_shape),
    #     n_blocks=args.num_blocks,
    #     intermediate_dims=args.dims,
    #     nonlinearity=args.nonlinearity,
    #     alpha=args.alpha,
    # )
    def make_ODETexture():
        odefunc = net.ODENet(args.dims, data_shape, args.strides, args.nonlinearity)
        return net.ODETexture(
            odefunc, T=1, solver=args.solver, atol=args.atol, rtol=args.rtol
        )

    model = nn.Sequential(
        *[make_ODETexture() for _ in range(args.num_blocks)],
        net.SigmoidTransform(args.alpha),
    )

    model = cvt(model)

    # training preconfig

    ## VGG features
    features = metrics.VGG19().to(device)
    features.load_state_dict(torch.load("vgg19.pth"))
    gmatrices_exemplar = list(map(metrics.GramMatrix, features(exemplar)))
    loss_fn = nn.MSELoss(reduction="mean")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_meter = utils.RunningAverageMeter(0.97)

    n_test_tex = 3
    test_noise = torch.randn(n_test_tex, 3, 512, 512, device=device)

    # training procedure
    with tqdm(total=args.num_epochs, desc="Epoch") as t:
        for ep in range(args.num_epochs):
            model.train()
            optimizer.zero_grad()

            ## generate some textures
            noise = torch.randn(args.batchsize, 3, 128, 128, device=device)
            generated_textures = model(noise)

            ## compute gram matrices
            gmatrices_samples = list(
                map(metrics.GramMatrix, features(generated_textures))
            )

            ## compute the gradients
            loss = 0.0
            for gmatrix_e, gmatrix_s in zip(gmatrices_exemplar, gmatrices_samples):
                loss += loss_fn(gmatrix_e.expand_as(gmatrix_s), gmatrix_s)

            loss.backward()

            optimizer.step()

            loss_meter.update(loss.item())

            if ep % args.num_disp_epochs == 0:
                with torch.no_grad():
                    model.eval()
                    tex = model(test_noise).to("cpu")
                    for i in range(n_test_tex):
                        tvio.write_png(
                            tforms.ConvertImageDtype(torch.uint8)(tex[i]),
                            opath.join(ws_path, f"tex_{ep}_{i}.png"),
                        )

                torch.save(
                    copy.deepcopy(model.state_dict()),
                    opath.join(ws_path, f"model_checkpoint.pth"),
                )

            t.set_postfix({"running_loss": f"{loss_meter.avg}"})
            t.update()
            ## test in some period
