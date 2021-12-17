from skimage.color.colorconv import xyz2luv
import torch
import torch.nn as nn

import torchvision as tv


def make_model(opt):
    return DRN(opt)

class DRN(nn.Module):
    def __init__(self, opt, num_classes=2):
        super(DRN, self).__init__()
        
        vgg = tv.models.vgg16_bn(pretrained=True)
        self.vgg_features = vgg.features
        self.num_ans_classes = num_classes

        # num_features = vgg.classifier[6].in_features
        self.num_features = 32768
        features = list(vgg.classifier.children())[1:-1] # Remove last layer
        features.insert(0, nn.Linear(self.num_features, 4096))
        features.extend([nn.Linear(4096, num_classes)]) # Add our layer with 4 outputs
        self.classifier = nn.Sequential(*features)


    def forward(self,x):
        x = self.vgg_features(x)
        out = self.classifier(x.view(-1, self.num_features))
        return out
