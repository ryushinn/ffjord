import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchdiffeq import odeint_adjoint as odeint

NONLINEARITIES = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "softplus": nn.Softplus(),
    "elu": nn.ELU(),
}

class ConcatConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)

class ODENet(nn.Module):

    """
    Convolutional ODE Net that defines the dynamics of image noise
    """

    def __init__(
        self, hidden_dims, init_dim, strides, nonlinearity="softplus"):
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
                layer_kwargs = {"ksize": 3, "stride": 1, "padding": 1, "transpose": False}
            elif stride == 2:
                layer_kwargs = {"ksize": 4, "stride": 2, "padding": 1, "transpose": False}
            elif stride == -2:
                layer_kwargs = {"ksize": 4, "stride": 2, "padding": 1, "transpose": True}
            else:
                raise ValueError('Unsupported stride: {}'.format(stride))

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

        if T is None: T = self.T
        assert T > 0, "The end time should be large than 0"

        integration_T = torch.tensor([0.0, T]).to(y)

        y_T = odeint(
            self.odefunc,
            y, integration_T,
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