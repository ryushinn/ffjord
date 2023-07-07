import argparse
import os
import time
import random
import numpy as np

import torch
import torch.optim as optim
import torchvision.io as tvio
import torchvision.datasets as dset
import torchvision.transforms as tforms
from torchvision.utils import save_image

import matplotlib.pyplot as plt
from tqdm import tqdm

import metrics
import lib.layers as layers
import lib.utils as utils
import lib.odenvp as odenvp
import lib.multiscale_parallel as multiscale_parallel

parser = argparse.ArgumentParser("ODE texture")

SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams']
parser.add_argument('--solver', type=str, default='dopri5', choices=SOLVERS)
parser.add_argument('--atol', type=float, default=1e-5)
parser.add_argument('--rtol', type=float, default=1e-5)

_converter = lambda s: tuple(map(int, s.split(",")))
parser.add_argument("--dims", type=_converter, default="8,32,32,8")
parser.add_argument("--strides", type=_converter, default="2,2,1,-2,-2")
parser.add_argument("--num_blocks", type=int, default=1, help='Number of stacked CNFs.')

parser.add_argument("--conv", type=eval, default=True, choices=[True, False])
parser.add_argument(
    "--layer_type", type=str, default="ignore",
    choices=["ignore", "concat", "concat_v2", "squash", "concatsquash", "concatcoord", "hyper", "blend"]
)
parser.add_argument(
    "--nonlinearity", type=str, default="softplus", choices=["tanh", "relu", "softplus", "elu", "swish"]
)

parser.add_argument("--alpha", type=float, default=1e-6)
parser.add_argument("--num_epochs", type=int, default=1000)
parser.add_argument("--num_disp_epochs", type=int, default=10)
parser.add_argument("--batchsize", type=int, default=10)
parser.add_argument("--lr", type=float, default=1e-3)

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
    exemplar = tforms.ConvertImageDtype(torch.float32)(tvio.read_image("grass_example_1.jpg"))
    exemplar = tforms.CenterCrop(32)(exemplar)
    tvio.write_png(tforms.ConvertImageDtype(torch.uint8)(exemplar), "cropped_exemplar.png")
    exemplar = cvt(exemplar)

    # model
    data_shape = exemplar.size()
    model = odenvp.ODENVP(
        (args.batchsize, *data_shape),
        n_blocks=args.num_blocks,
        intermediate_dims=args.dims,
        nonlinearity=args.nonlinearity,
        alpha=args.alpha,
    )

    print(model)
    
    # training preconfig
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_meter = utils.RunningAverageMeter(0.97)
    exemplar = torch.unsqueeze(exemplar, 0) # add the batch dim

    n_test_tex = 1
    test_noise = torch.randn(n_test_tex, *data_shape, device=device)
    
    # training procedure
    with tqdm(total=args.num_epochs, desc="Epoch") as t:
        for ep in range(args.num_epochs):
            model.train()
            optimizer.zero_grad()

            ## generate some textures
            noise = torch.randn(args.batchsize, *data_shape, device=device)
            generated_textures = model(noise, reverse=True).view(-1, *data_shape)
            
            ## compute the gradients
            loss = metrics.VGGLoss(exemplar, generated_textures)
            loss.backward()

            optimizer.step()

            loss_meter.update(loss.item())

            if ep % args.num_disp_epochs == 0:
                with torch.no_grad():
                    model.eval()
                    tex = model(test_noise, reverse=True).view(-1, *data_shape).to("cpu")
                    for i in range(n_test_tex):
                        remapped = (tex[i] - tex[i].min())/(tex[i].max() - tex[i].min())
                        tvio.write_png(
                            tforms.ConvertImageDtype(torch.uint8)(remapped),
                            f"tex_{ep}_{i}.png"
                        )

            t.set_postfix({"running_loss": f"{loss_meter.avg}"})
            t.update()
            ## test in some period
