import torch
import torch.nn as nn

import logging
logger = logging.getLogger(__name__)

class DebugLayer(nn.Module):
    def __init__(self, name: str|None = None):
        super().__init__()
        self.name = None
        if name:
            self.name = name
        
    def forward(self, x):
        if self.name:
            logger.debug(f"Input shape: {x.shape} in {self.name}")
        else:
            logger.debug(f"Input shape: {x.shape}")
        if not type(x) == torch.Tensor:
            raise TypeError(f"Input must be of type torch.Tensor, not {type(x)}")
        return x

class LinearBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, relu = nn.ReLU) -> None:
        super().__init__()
        self.debug = DebugLayer(name=self.__class__.__name__)
        self.linear = nn.Linear(in_channels, out_channels)
        self.relu = relu()
        
    def forward(self, x: torch.Tensor):
        x = self.debug(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = self.relu(x)
        #x = self.bn(x)
        return x
    
class SingleConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel= 3,stride: int = 1,padding = 0, activation = nn.ReLU, norm = True) -> None:
        super().__init__()
        self.body = nn.Sequential(
            DebugLayer(name=self.__class__.__name__),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride= stride,padding=padding),
            activation(),
            nn.BatchNorm2d(out_channels) if norm else nn.Identity()
        )
        
    def forward(self, x: torch.Tensor):
        return self.body(x)
    
class SingleUpConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel= 3,stride: int = 1, padding = 0, activation = nn.ReLU) -> None:
        super().__init__()
        
        self.body = nn.Sequential(
            DebugLayer(name=self.__class__.__name__),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel, stride= stride, padding=padding),
            activation(),
            nn.BatchNorm2d(out_channels)
        )
        
    def forward(self, x: torch.Tensor):
        return self.body(x)
  
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride=1, padding=1, activation = nn.ELU):
        super().__init__()

        self.debug = DebugLayer(name=self.__class__.__name__)
        self.maxpool = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                               stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = activation(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        x = self.debug(x)
        x = self.maxpool(x)

        out = self.relu(self.bn1(self.conv1(x)))
        
        out = self.bn2(self.conv2(out))

        out += self.shortcut(x)

        out = self.relu(out)
        
        return out


            