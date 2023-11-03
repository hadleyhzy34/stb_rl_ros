from os import device_encoding
import matplotlib.pyplot as plt
from gymnasium import spaces
import math
import pdb
import torch
import numpy as np
from torch import nn
import torchvision.models as models
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from agent.fasternet import FasterNet
from util.costmap import state2costmap


class CustomFeature(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        # self.feat = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        # self.feat = models.efficientnet_v2_s(weights=None)
        # self.feature_extractor = torch.nn.Sequential(*(list(self.feat.children())[0][:-1]))
        # self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        # self.linear = nn.Sequential(nn.Linear(1000, features_dim), nn.ReLU())
        self.feature_extractor = FasterNet(3,features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        observations = state2costmap(observations)

        # pdb.set_trace()
        output = self.feature_extractor.forward_cls(observations)
        # output = self.linear(self.feat(observations))
        # output = self.pool(self.feature_extractor(observations))[:,:,0,0]
        return output
        # return self.linear(self.feature_extractor(observations))

if __name__ == "__main__":
    pdb.set_trace()
    # resnext50_32x4d = models.resnext50_32x4d(pretrained=True)
    # test = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
    # print(resnext50_32x4d)
    input = torch.rand(4,362)*4.
    output = state2costmap(input)
