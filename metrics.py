import torch
import torch.nn as nn

from vgg import vgg19

class VGG19(torch.nn.Module):

    def __init__(self):
        super(VGG19, self).__init__()

        self.block1_conv1 = torch.nn.Conv2d(3, 64, (3,3), padding=(1,1), padding_mode='reflect')
        self.block1_conv2 = torch.nn.Conv2d(64, 64, (3,3), padding=(1,1), padding_mode='reflect')

        self.block2_conv1 = torch.nn.Conv2d(64, 128, (3,3), padding=(1,1), padding_mode='reflect')
        self.block2_conv2 = torch.nn.Conv2d(128, 128, (3,3), padding=(1,1), padding_mode='reflect')

        self.block3_conv1 = torch.nn.Conv2d(128, 256, (3,3), padding=(1,1), padding_mode='reflect')
        self.block3_conv2 = torch.nn.Conv2d(256, 256, (3,3), padding=(1,1), padding_mode='reflect')
        self.block3_conv3 = torch.nn.Conv2d(256, 256, (3,3), padding=(1,1), padding_mode='reflect')
        self.block3_conv4 = torch.nn.Conv2d(256, 256, (3,3), padding=(1,1), padding_mode='reflect')

        self.block4_conv1 = torch.nn.Conv2d(256, 512, (3,3), padding=(1,1), padding_mode='reflect')
        self.block4_conv2 = torch.nn.Conv2d(512, 512, (3,3), padding=(1,1), padding_mode='reflect')
        self.block4_conv3 = torch.nn.Conv2d(512, 512, (3,3), padding=(1,1), padding_mode='reflect')
        self.block4_conv4 = torch.nn.Conv2d(512, 512, (3,3), padding=(1,1), padding_mode='reflect')

        self.relu = torch.nn.ReLU(inplace=True)
        self.downsampling = torch.nn.AvgPool2d((2,2))

    def forward(self, image):
        
        # RGB to BGR
        image = image[:, [2,1,0], :, :]

        # [0, 1] --> [0, 255]
        image = 255 * image

        # remove average color
        image[:,0,:,:] -= 103.939
        image[:,1,:,:] -= 116.779
        image[:,2,:,:] -= 123.68

        # block1
        block1_conv1 = self.relu(self.block1_conv1(image))
        block1_conv2 = self.relu(self.block1_conv2(block1_conv1))
        block1_pool = self.downsampling(block1_conv2)

        # block2
        block2_conv1 = self.relu(self.block2_conv1(block1_pool))
        block2_conv2 = self.relu(self.block2_conv2(block2_conv1))
        block2_pool = self.downsampling(block2_conv2)

        # block3
        block3_conv1 = self.relu(self.block3_conv1(block2_pool))
        block3_conv2 = self.relu(self.block3_conv2(block3_conv1))
        block3_conv3 = self.relu(self.block3_conv3(block3_conv2))
        block3_conv4 = self.relu(self.block3_conv4(block3_conv3))
        block3_pool = self.downsampling(block3_conv4)

        # block4
        block4_conv1 = self.relu(self.block4_conv1(block3_pool))
        block4_conv2 = self.relu(self.block4_conv2(block4_conv1))
        block4_conv3 = self.relu(self.block4_conv3(block4_conv2))
        block4_conv4 = self.relu(self.block4_conv4(block4_conv3))

        return [block1_conv1, block1_conv2, block2_conv1, block2_conv2, block3_conv1, block3_conv2, block3_conv3, block3_conv4, block4_conv1, block4_conv2, block4_conv3, block4_conv4]


class VGGFeatures(nn.Module):
    def __init__(self):
        super(VGGFeatures, self).__init__()

        vgg_pretrained_features = vgg19(pretrained=True).features

        self.features = vgg_pretrained_features

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        ## normalize
        # mean = [0.485, 0.456, 0.406]
        # std = [0.229, 0.224, 0.225]
        # device = "cuda" if x.is_cuda else "cpu"
        # mean = torch.as_tensor(mean, device=device).view(1, 3, 1, 1)
        # std = torch.as_tensor(std, device=device).view(1, 3, 1, 1)
        # x = x.sub(mean)
        # x = x.div(std)

        # get features

        # important_layers = [
        #     1, 3, # block1_conv1, 2
        #     6, 8, # block2_conv1, 2
        #     11, 13, 15, 17, # block3_conv1, 2, 3, 4
        #     20, 22, 24, 26, # block4_conv1, 2, 3, 4
        #     29, 31, 33, 35, # block5_conv1, 2, 3, 4
        # ]

        important_layers = [3, 8, 17, 26, 35]  # only the last conv of each block

        features_layers = []
        for i in range(len(self.features)):
            x = self.features[i](x)

            if i in important_layers:
                features_layers.append(x)

        return features_layers


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
