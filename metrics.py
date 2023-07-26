import torch
import torch.nn as nn

from vgg import vgg19


class VGGFeatures(nn.Module):
    def __init__(self):
        super(VGGFeatures, self).__init__()

        vgg_pretrained_features = vgg19(pretrained=True).features

        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()

        for x in range(4):  # relu_1_1
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):  # relu_2_1
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 18):  # relu_3_1
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(18, 27):  # relu_4_1
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(27, 36):  # relu_5_1
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        ## normalize
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        device = "cuda" if x.is_cuda else "cpu"
        mean = torch.as_tensor(mean, device=device).view(1, 3, 1, 1)
        std = torch.as_tensor(std, device=device).view(1, 3, 1, 1)
        x = x.sub(mean)
        x = x.div(std)

        # get features
        h1 = self.slice1(x)
        h_relu1_1 = h1
        h2 = self.slice2(h1)
        h_relu2_1 = h2
        h3 = self.slice3(h2)
        h_relu3_1 = h3
        h4 = self.slice4(h3)
        h_relu4_1 = h4
        h5 = self.slice5(h4)
        h_relu5_1 = h5

        return [h_relu1_1, h_relu2_1, h_relu3_1, h_relu4_1, h_relu5_1]


def GramMatrix(input):
    b, c, h, w = input.size()
    features = input.view(b, c, h * w)
    gram_matrix = torch.bmm(features, features.transpose(1, 2))

    gram_matrix.div_(h * w)
    return gram_matrix


def align_size(a: torch.Tensor, b: torch.Tensor):
    b1, c, n1 = a.size()
    b2, c, n2 = b.size()

    if b1 != b2:
        if b1 == 1:
            a = a.repeat(b2, 1, 1)
        elif b2 == 1:
            b = b.repeat(b1, 1, 1)
        else:
            raise ValueError(f"cannot broadcast between {b1} and {b2} size")

    if n1 != n2:
        if n1 < n2:
            a = a.repeat(1, 1, n2 // n1)
            indices = torch.randint(0, n1, (n2 % n1,))
            a = torch.concat([a, a[:, :, indices]], dim=2)
        else:
            b = b.repeat(1, 1, n1 // n2)
            indices = torch.randint(0, n2, (n1 % n2,))
            b = torch.concat([b, b[:, :, indices]], dim=2)

    assert a.size() == b.size()

    return a, b


def SlicedWassersteinLoss(features1, features2):
    assert len(features1) == len(features2)

    loss = 0.0
    for f1, f2 in zip(features1, features2):
        b1, c1, h1, w1 = f1.size()
        b2, c2, h2, w2 = f2.size()
        assert c1 == c2

        f1 = f1.view(b1, c1, -1)
        f2 = f2.view(b2, c2, -1)

        # align the size
        f1, f2 = align_size(f1, f2)

        # get c random directions
        Vs = torch.randn(c1, c1).to(f1)
        Vs = Vs / torch.sqrt(torch.sum(Vs**2, dim=1, keepdim=True))

        # project
        pf1 = torch.einsum("bcn,mc->bmn", f1, Vs)
        pf2 = torch.einsum("bcn,mc->bmn", f2, Vs)

        # sort
        spf1 = torch.sort(pf1, dim=2)[0]
        spf2 = torch.sort(pf2, dim=2)[0]

        # MSE
        loss += torch.mean((spf1 - spf2) ** 2)

    return loss
