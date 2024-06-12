import torch.nn as nn
import torch

class PartGlobalAveragePooling(nn.Module):

    def __init__(self, part_num):
        super(PartGlobalAveragePooling, self).__init__()
        self.avgpool_c = nn.AdaptiveAvgPool2d((part_num, 1))
        dropout = nn.Dropout(p=0.5)
        self.maxpool_g = nn.AdaptiveAvgPool2d((1, 1))

        self.pool_c = nn.Sequential(self.avgpool_c, dropout)
        self.pool_g = nn.Sequential(self.maxpool_g)

    def forward(self, features):
        features_part = torch.squeeze(self.pool_c(features))
        features_global = torch.squeeze(self.pool_g(features))
        return features_part, features_global