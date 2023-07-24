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


def align_size(a, b):
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
            indices = torch.randint(0, n1, (n2 - n1,))
            a = torch.concat([a, a[:, :, indices]], dim=2)
        else:
            indices = torch.randint(0, n2, (n1 - n2,))
            b = torch.concat([b, b[:, :, indices]], dim=2)

    return a, b


def SlicedWassersteinLoss(features1, features2):
    assert len(features1) == len(features2)

    loss = 0.0
    for l1, l2 in zip(features1, features2):
        b1, c1, h1, w1 = features1.size()
        b2, c2, h2, w2 = features2.size()
        assert c1 == c2

        features1 = features1.view(b1, c1, -1)
        features2 = features2.view(b2, c2, -1)

        # align the size
        features1, features2 = align_size(features1, features2)

        # get c random directions
        Vs = torch.randn(c1, c1).to(features1)
        Vs = Vs / torch.sqrt(torch.sum(Vs**2, dim=1, keepdim=True))

        # project
        pfeatures1 = torch.einsum("bcn,mc->bnm", features1, Vs)
        pfeatures2 = torch.einsum("bcn,mc->bnm", features2, Vs)

        # sort
        spfeatures1 = torch.sort(pfeatures1, dim=2)[0]
        spfeatures2 = torch.sort(pfeatures2, dim=2)[0]

        # MSE
        loss += torch.mean((spfeatures1 - spfeatures2) ** 2)

    return loss
