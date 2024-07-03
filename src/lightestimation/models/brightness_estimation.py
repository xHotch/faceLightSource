# The Models in this file are based on works by Sztrajman et al. [1] and Weber et al. [2].
# 
# [1] Sztrajman, A., Neophytou, A., Weyrich, T., & Sommerlade, E. (2020). High-Dynamic-Range Lighting Estimation From Face Portraits. 2020 International Conference on 3D Vision (3DV), 355–363. https://doi.org/10.1109/3DV50981.2020.00045
# [2] Weber, H., Prévost, D., & Lalonde, J.-F. (2018). Learning to Estimate Indoor Lighting from 3D Objects (arXiv:1806.03994). arXiv. http://arxiv.org/abs/1806.03994

from typing import Any, Mapping
import torch
import torch.nn as nn

import logging

from lightestimation.models.blocks import LinearBlock, SingleConvBlock, SingleUpConvBlock, ResidualBlock

logger = logging.getLogger(__name__)

class Env2EnvAutoEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, bottleneck_size: int = 256, activation: type[nn.Module] = nn.LeakyReLU, relu_output: bool = False) -> None:
        super().__init__()
        
        self.encoder = nn.Sequential(
            SingleConvBlock(in_channels, 96, kernel=3, stride=1, padding=1, activation = activation),
            nn.MaxPool2d(2),
            SingleConvBlock(96, 96, kernel=3, stride=1, padding=1, activation = activation),
            nn.MaxPool2d(2),
            SingleConvBlock(96, 64, kernel=3, stride=1, padding=1, activation = activation),
            nn.MaxPool2d(2),
            LinearBlock(32*64*64, bottleneck_size, nn.Identity)
        )
        
        self.decoder = Decoder(out_channels=in_channels, bottleneck_size=bottleneck_size, activation = activation, relu_output=relu_output)
        
    def forward(self, x) -> torch.Tensor:
        x = self.encoder(x)
        x_out = self.decoder(x)
        return x_out

class Decoder(nn.Module):
    def __init__(self, out_channels: int, bottleneck_size: int, activation: type[nn.Module], relu_output: bool) -> None:
        super().__init__()
        self.relu_output = nn.ReLU if relu_output else nn.Identity
        self.middle_part = LinearBlock(bottleneck_size, 32*64*16)
        self.decoder = nn.Sequential(
            SingleUpConvBlock(16, 64, kernel=4, stride=2, padding=1, activation = activation),
            SingleUpConvBlock(64, 96, kernel=4, stride=2, padding=1, activation = activation),
            SingleUpConvBlock(96, 96, kernel=4, stride=2, padding=1, activation = activation),
            SingleConvBlock(96, 64, kernel=3, stride=1, padding=1,activation = activation),
            SingleConvBlock(64, out_channels, kernel=3, stride=1, padding=1, activation=self.relu_output, norm=False),
        )
    
    def forward(self, x) -> torch.Tensor:
        x_middle = self.middle_part(x)
        x_middle = x_middle.view(x_middle.size(0), 16, 32, 64)
        x_out = self.decoder(x_middle)
        return x_out
    
class Face2EnvNetwork(nn.Module):
    def __init__(self, in_channels:int=1, out_channels:int=3, decoder_state_dict:Mapping[str, Any]|None = None, bottleneck_size:int=256, activation: type[nn.Module] = nn.LeakyReLU, initial_feature_maps:int=16, train_decoder:bool=False) -> None:
        super().__init__()
        self.face2latent = FaceEncoder(in_channels=in_channels, bottleneck_size=bottleneck_size, activation=activation, initial_feature_maps=initial_feature_maps)
        self.decoder = Decoder(out_channels=out_channels, bottleneck_size=bottleneck_size, activation = activation, relu_output=True)
        if decoder_state_dict:
            self.decoder.load_state_dict(decoder_state_dict)
            logger.debug("Loaded decoder weights")

        if not train_decoder:
            self.decoder.requires_grad_(False)
        
    def forward(self, x) -> torch.Tensor:
        x = self.face2latent(x)
        x = self.decoder(x)
        return x
        
class FaceEncoder(nn.Module):
    def __init__(self, in_channels:int=1, bottleneck_size:int=256, activation: type[nn.Module] = nn.LeakyReLU, initial_feature_maps:int=16) -> None:
        super().__init__()
        self.face2latent=nn.Sequential(
            SingleConvBlock(in_channels, initial_feature_maps, kernel=3, stride=1, padding=1, activation=activation), 
            ResidualBlock(initial_feature_maps, initial_feature_maps, stride=1, kernel_size=3, activation=activation), 
            ResidualBlock(initial_feature_maps, initial_feature_maps*2, stride=1, kernel_size=3, activation=activation),
            ResidualBlock(initial_feature_maps*2, initial_feature_maps*4, stride=1, kernel_size=3, activation=activation),
            ResidualBlock(initial_feature_maps*4, initial_feature_maps, activation=activation), 
            LinearBlock(initial_feature_maps*16*16, bottleneck_size, relu=nn.Identity),
        )
        
    def forward(self, x) -> torch.Tensor:
        x = self.face2latent(x)
        return x