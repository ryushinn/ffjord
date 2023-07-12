import torch
import torch.nn as nn

from vgg import vgg19


class VGGFeatures(nn.Module):

    def __init__(self):
        super(VGGFeatures, self).__init__()

        vgg_pretrained_features = vgg19(pretrained=True).features
        
        self.features = vgg_pretrained_features

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
        important_layers = [
            1, 3, # block1_conv1, 2
            6, 8, # block2_conv1, 2
            11, 13, 15, 17, # block3_conv1, 2, 3, 4
            20, 22, 24, 26, # block4_conv1, 2, 3, 4
            29, 31, 33, 35, # block5_conv1, 2, 3, 4
        ]

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