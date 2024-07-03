import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LRScheduler
from prodigyopt import Prodigy # type: ignore

import logging

logger = logging.getLogger(__name__)


class ModelWrapper():
    device: torch.device
    
    def __init__(self, net: nn.Module, loss_fn: nn.Module|None = None, lr:float = 1.0, config: dict = {}) -> None:
        self.net = net
        self.loss_fn = loss_fn() if loss_fn else None 

        weight_decay = 0.01
        if "weight_decay" in config:
            weight_decay = config["weight_decay"]
        self.optimizer = Prodigy(self.net.parameters(), lr=lr, weight_decay=weight_decay)

        self.scheduler: LRScheduler|None = None
        if "scheduler" in config:
            self.scheduler = config["scheduler"]
            
        if "cosine_annealing" in config and config["cosine_annealing"]:
            epochs = config["epochs"]
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        
    def update_device(self, device: str|torch.device) -> None:
        if type(device) == str:
            self.device = torch.device(device)
        elif type(device) == torch.device:
            self.device = device
        else:
            raise ValueError("Device must be a string or torch.device")
        self.net.to(self.device)
    
    def train(self, data: torch.Tensor, label: torch.Tensor, *args, **kwargs) -> float:
        try:
            self.net.train()
            self.optimizer.zero_grad()
            data = data.to(self.device)
            label = label.to(self.device)
            
            output = self.net(data)
            if not self.loss_fn:
                raise RuntimeError("No loss function defined")
            else:
                loss = self.loss_fn(output, label, *args, **kwargs)
            loss.backward()
            
            self.optimizer.step()
            
            return loss.item()
        except Exception as e:
            raise RuntimeError(f"Error during training: {e}")
        
    def update_schedule(self, *args, **kwargs) -> None:
        if self.scheduler is not None:
            self.scheduler.step(*args, **kwargs)
        
    def predict(self, data: torch.Tensor) -> torch.Tensor:
        try:
            self.net.eval()
            data = data.to(self.device)
            with torch.no_grad():
                output = self.net(data)
                return output
        except Exception as e:
            raise RuntimeError(f"Error during prediction: {e}")
    
def predict_model(subnet: nn.Module, data, device) -> torch.Tensor:
    try:
        subnet.eval()
        data = data.to(device)
        with torch.no_grad():
            output = subnet(data)
            return output
    except Exception as e:
        raise RuntimeError(f"Error during prediction: {e}")