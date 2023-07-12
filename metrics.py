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
        device = 'cuda' if x.is_cuda else 'cpu'
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