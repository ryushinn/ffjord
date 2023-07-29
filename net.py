import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchdiffeq import odeint_adjoint as odeint
from diffusers import UNet2DModel

NONLINEARITIES = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "softplus": nn.Softplus(),
    "elu": nn.ELU(),
}


class ConcatConv2d(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        ksize=3,
        stride=1,
        padding=0,
        padding_mode="circular",
        dilation=1,
        groups=1,
        bias=True,
        transpose=False,
    ):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1,
            dim_out,
            kernel_size=ksize,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)


class ODENet(nn.Module):

    """
    Convolutional ODE Net that defines the dynamics of image noise
    """

    def __init__(self, hidden_dims, init_dim, strides, nonlinearity="softplus"):
        super(ODENet, self).__init__()

        assert len(strides) == len(hidden_dims) + 1
        base_layer = ConcatConv2d

        # build layers and add them
        layers = []
        activation_fns = []
        hidden_shape = init_dim

        for dim_out, stride in zip(hidden_dims + (init_dim,), strides):
            if stride is None:
                layer_kwargs = {}
            elif stride == 1:
                layer_kwargs = {
                    "ksize": 3,
                    "stride": 1,
                    "padding": 1,
                    "transpose": False,
                }
            elif stride == 2:
                layer_kwargs = {
                    "ksize": 4,
                    "stride": 2,
                    "padding": 1,
                    "transpose": False,
                }
            elif stride == -2:
                layer_kwargs = {
                    "ksize": 4,
                    "stride": 2,
                    "padding": 1,
                    "transpose": True,
                }
            else:
                raise ValueError("Unsupported stride: {}".format(stride))

            layer = base_layer(hidden_shape, dim_out, **layer_kwargs)
            layers.append(layer)
            activation_fns.append(NONLINEARITIES[nonlinearity])

            hidden_shape = dim_out

        # zero init the last layer to make ode identity
        with torch.no_grad():
            for param in layers[-1].parameters():
                nn.init.zeros_(param)

        self.layers = nn.ModuleList(layers)
        self.activation_fns = nn.ModuleList(activation_fns[:-1])

    def forward(self, t, y):
        dx = y
        for l, layer in enumerate(self.layers):
            dx = layer(t, dx)
            # if not last layer, use nonlinearity
            if l < len(self.layers) - 1:
                dx = self.activation_fns[l](dx)
        return dx


class ODETexture(nn.Module):
    def __init__(self, odefunc, T=1.0, solver="dopri5", atol=1e-5, rtol=1e-5) -> None:
        super(ODETexture, self).__init__()

        self.odefunc = odefunc
        self.T = T
        self.solver = solver
        self.atol = atol
        self.rtol = rtol

    def forward(self, y, T=None):
        if T is None:
            T = self.T
        assert T > 0, "The end time should be large than 0"

        integration_T = torch.tensor([0.0, T]).to(y)

        y_T = odeint(
            self.odefunc,
            y,
            integration_T,
            rtol=self.rtol,
            atol=self.atol,
            method=self.solver,
        )

        return y_T[1]


class SigmoidTransform(nn.Module):
    """Reverse of LogitTransform."""

    def __init__(self, alpha):
        nn.Module.__init__(self)
        self.alpha = alpha

    def forward(self, x):
        return _sigmoid(x, self.alpha)


def _sigmoid(x, alpha):
    y = (torch.sigmoid(x) - alpha) / (1 - alpha)
    return y


class HiddenUnits(nn.Module):
    def __init__(self, units):
        super().__init__()

        assert isinstance(units, list)

        self.units = nn.ModuleList(units)

    def forward(self, t, x):
        dx = 0.0
        for unit in self.units:
            dx += unit(t, x)

        return dx


class UNet(nn.Module):
    def __init__(self, inout_channels, channels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        down_block_types = tuple("DownBlock2D" for _ in range(len(channels)))
        up_block_types = tuple("UpBlock2D" for _ in range(len(channels)))

        self.network = UNet2DModel(
            out_channels=inout_channels,
            in_channels=inout_channels,
            # arch
            block_out_channels=channels,
            up_block_types=up_block_types,
            down_block_types=down_block_types,
            layers_per_block=2,
            add_attention=False,
            # time embedding
            time_embedding_type="positional",
            freq_shift=0,
            flip_sin_to_cos=False,
        )

        for module in self.network.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                module.padding_mode = "circular"

    def forward(self, t, x):
        return self.network(x, t)["sample"]


def get_kernel(size=5, channels=3):
    # TODO: real kernel size
    assert size % 2 == 1, "kernel size must be odd!"

    k1d = torch.tensor([1, 4, 6, 4, 1])
    k2d = torch.outer(k1d, k1d)
    k2d = k2d / k2d.sum()

    k2d = k2d[None, None, :, :].repeat(channels, 1, 1, 1)

    assert k2d.size() == (channels, 1, size, size)
    return k2d


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs)


downsample = nn.AvgPool2d(2, 2)
upsample = nn.Upsample(scale_factor=2)


def rand_pyramid(size: list[int], num_layers, device):
    b, c, h, w = size

    pyramid = ()
    for i in range(num_layers):
        pyramid += (
            torch.randn(
                b,
                c,
                h // (2**i),
                w // (2**i),
                device=device,
            ),
        )
    pyramid += (
        torch.randn(
            b,
            c,
            h // (2**num_layers),
            w // (2**num_layers),
            device=device,
        ),
    )

    return pyramid


def encoder(
    x: torch.Tensor,
    num_levels: int = 3,
) -> tuple[torch.Tensor]:
    # the size of x is b, c, h, w
    assert len(x.size()) == 4, "the input tensor should be in (B, C, H, W) format"

    ksize = 5
    kernel = get_kernel(ksize, x.size(1)).to(x)
    padding = ksize // 2

    pyramid = ()
    current_level = x
    for i in range(num_levels):
        # gaussian filter x
        x_blur = F.conv2d(
            F.pad(current_level, (padding,) * 4, mode="circular"),
            kernel,
            stride=1,
            padding=0,
            groups=x.size(1),
        )

        # downsample x
        x_down = downsample(x_blur)

        # upsample x
        x_up = upsample(x_down)

        # compute the difference
        laplacian = current_level - x_up

        # append difference
        current_level = x_down

        pyramid += (laplacian,)

    # append the remainder
    pyramid += (current_level,)

    return pyramid


def decoder(pyramid: tuple[torch.Tensor]) -> torch.Tensor:
    current_level = pyramid[-1]

    for level in reversed(pyramid[:-1]):
        current_level = upsample(current_level)
        current_level += level

    return current_level


def compile(pyramid: tuple[torch.Tensor]) -> torch.Tensor:
    levels = []
    for i, laplacian in enumerate(pyramid):
        levels.append(nn.Upsample(scale_factor=2**i)(laplacian))

    return torch.concat(levels, dim=1)


def decompile(x: torch.Tensor, channels=3) -> tuple[torch.Tensor]:
    assert x.size(1) % channels == 0

    levels = ()
    for i, level in enumerate(torch.chunk(x, x.size(1) // channels, dim=1)):
        levels += (nn.AvgPool2d(2**i, 2**i)(level),)

    return levels
