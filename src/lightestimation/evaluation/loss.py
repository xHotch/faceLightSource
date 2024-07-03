import torch
import torch.nn as nn

import numpy as np

from lightestimation.evaluation.accuracy import get_solid_angle_weights

class L2Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fn = nn.MSELoss(reduction="sum")
        
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.fn(prediction, target)

class LogL1Loss_solid_angle(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.is_weight_initialized = False
        self.weight: torch.Tensor
            
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not self.is_weight_initialized:
            self.is_weight_initialized = True
            self.weight = get_solid_angle_weights(prediction)
        device = prediction.device
        self.weight = self.weight.to(device)
        intermed = self.weight*(torch.log10(target+1) - torch.log10(prediction+1))
        return torch.sum(torch.abs(intermed))

class L1Loss_solid_angle(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.is_weight_initialized = False
        self.weight: torch.Tensor
            
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not self.is_weight_initialized:
            self.is_weight_initialized = True
            self.weight = get_solid_angle_weights(prediction)
        device = prediction.device
        self.weight = self.weight.to(device)
        return torch.sum(torch.abs(self.weight*(target - prediction)))
    
    
if __name__ == "__main__":
    all_metrics = [L1Loss_solid_angle(), LogL1Loss_solid_angle(), L2Loss()]    
    true = torch.ones(20,3, 256, 512)
    pred = torch.ones(20, 3, 256, 512) + 40
    for metric in all_metrics:
        try:
            value = metric(true, pred)
            print(f"{metric.__class__.__name__}: {value}")
        except Exception as e:
            print(f"Error in {metric.__class__.__name__}: {e}")
        