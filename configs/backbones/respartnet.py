import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class ResPartNet(nn.Module):

    def __init__(self, part_num):
        super(ResPartNet, self).__init__()

        # attributes
        self.part_num = part_num

        # backbone and optimize its architecture
        resnet = torchvision.models.resnet50(pretrained=True)
        resnet.layer4[0].downsample[0].stride = (1,1)
        resnet.layer4[0].conv2.stride = (1,1)

        # cnn feature
        self.resnet_conv = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)


    def forward(self, x):

        features = self.resnet_conv(x)
        # features_c = torch.squeeze(self.pool_c(features))
        # features_e = torch.squeeze(self.pool_e(features))
        return features

