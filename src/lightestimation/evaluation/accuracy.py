import numpy as np
import torch
import torch.nn as nn
from envmap import EnvironmentMap # type: ignore


def get_solid_angle_weights(size_tensor: torch.Tensor) -> torch.Tensor:
    output_size = size_tensor.shape[2:4]
    solid_angle_envmap = EnvironmentMap(np.zeros((output_size[0], output_size[1], 3)), "latlong")
    weight: torch.Tensor = torch.from_numpy(solid_angle_envmap.solidAngles())
    return weight

class PerformanceMetric(nn.Module):
    def __init__(self) -> None:
        self.values: list
        super().__init__()
                
    def reset_values(self, batch_index: int) -> None:
        if batch_index == 0:
            self.values = []
            
class MSE_solidangle(PerformanceMetric):
    def __init__(self) -> None:
        super().__init__()
        
        self.values: list[torch.Tensor] = []
        self.name = self.__class__.__name__
        self.is_weight_initialized = False
        self.weight: torch.Tensor
        
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, i) -> torch.Tensor:
        self.reset_values(i)
        if not self.is_weight_initialized:
            self.is_weight_initialized = True
            self.weight = get_solid_angle_weights(y_true)
        
        device = y_true.device
        self.weight = self.weight.to(device) 
        mean = torch.mean(self.weight*torch.pow((y_true - y_pred),2), dim=(1,2,3)).to("cpu")
        self.values.append(mean)
        return mean
    
class LOG_MSE_solidangle(PerformanceMetric):
    def __init__(self) -> None:
        super().__init__()
        
        self.values: list[torch.Tensor] = []
        self.name = self.__class__.__name__
        self.is_weight_initialized = False
        self.weight: torch.Tensor
        
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, i) -> torch.Tensor:
        self.reset_values(i)
        if not self.is_weight_initialized:
            self.is_weight_initialized = True
            self.weight = get_solid_angle_weights(y_true)
        
        device = y_true.device
        self.weight = self.weight.to(device)
        if not self.is_weight_initialized:
            self.is_weight_initialized = True
            self.weight = get_solid_angle_weights(y_true)
        mean = torch.mean(self.weight*torch.pow((torch.log(y_true+1) - torch.log(y_pred+1)),2), dim=(1,2,3)).to("cpu")
        self.values.append(mean)
        return mean    
    
class MSE(PerformanceMetric):
    def __init__(self) -> None:
        super().__init__()
        
        self.values: list[torch.Tensor] = []

        self.name = self.__class__.__name__
        self.fn = torch.nn.MSELoss(reduction="none")
        
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, i) -> torch.Tensor:
        if y_pred.shape==4:
            dim = [1,2,3]
        else:
            dim = [1]
        self.reset_values(i)
        value = self.fn(y_pred, y_true)
        value = torch.mean(value, dim=dim).to("cpu")
        self.values.append(value)
        return value
    
class RMSE(PerformanceMetric):
    def __init__(self) -> None:
        super().__init__()
        
        self.values: list[torch.Tensor] = []

        self.name = self.__class__.__name__
        self.fn = torch.nn.MSELoss(reduction="none")
        
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, i) -> torch.Tensor:
        if y_pred.shape==4:
            dim = [1,2,3]
        else:
            dim = [1]
        self.reset_values(i)
        value = self.fn(y_pred, y_true)
        value = torch.mean(value, dim=dim)
        value = torch.sqrt(value).to("cpu")
        self.values.append(value)
        return value
    
class MAE(PerformanceMetric):
    def __init__(self) -> None:
        super().__init__()
        
        self.values: list[torch.Tensor] = []

        self.name = self.__class__.__name__
        self.fn = torch.nn.L1Loss(reduction="none")
        
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, i) -> torch.Tensor:
        if y_pred.shape==4:
            dim = [1,2,3]
        else:
            dim = [1]
        self.reset_values(i)
        value = self.fn(y_pred, y_true)
        value = torch.mean(value, dim=dim).to("cpu")
        self.values.append(value)
        return value
         
class MAE_solidangle(PerformanceMetric):
    def __init__(self) -> None:
        super().__init__()
        self.values: list[torch.Tensor] = []
        self.name = self.__class__.__name__
        self.is_weight_initialized = False
        self.weight: torch.Tensor
        self.fn = torch.nn.L1Loss(reduction="none")
        
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, i) -> torch.Tensor:
        self.reset_values(i)
        if not self.is_weight_initialized:
            self.is_weight_initialized = True
            self.weight = get_solid_angle_weights(y_true)
        device = y_true.device
        self.weight = self.weight.to(device)    
        value = torch.mean(self.weight*torch.abs((y_true - y_pred)), dim=(1,2,3)).to("cpu")
        self.values.append(value)
        return value
    
class LOG_MAE_solidangle(PerformanceMetric):
    def __init__(self) -> None:
        super().__init__()
        self.values: list[torch.Tensor] = []
        self.name = self.__class__.__name__
        self.is_weight_initialized = False
        self.weight: torch.Tensor
        self.fn = torch.nn.L1Loss(reduction="none")
        
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, i) -> torch.Tensor:
        self.reset_values(i)
        if not self.is_weight_initialized:
            self.is_weight_initialized = True
            self.weight = get_solid_angle_weights(y_true)
        device = y_true.device
        self.weight = self.weight.to(device)    
        value = torch.mean(self.weight*torch.abs((torch.log(y_true+1) - torch.log(y_pred+1))), dim=(1,2,3)).to("cpu")
    
        self.values.append(value)
        return value

class RMSE_solidangle(PerformanceMetric):
    def __init__(self) -> None:
        super().__init__()
        
        self.values: list[torch.Tensor] = []
        self.name = self.__class__.__name__
        self.is_weight_initialized = False
        self.weight: torch.Tensor
        
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, i) -> torch.Tensor:
        self.reset_values(i)
        if not self.is_weight_initialized:
            self.is_weight_initialized = True
            self.weight = get_solid_angle_weights(y_true)
        device = y_true.device
        self.weight = self.weight.to(device)   
        mean = torch.mean(self.weight*torch.pow((y_true - y_pred),2), dim=(1,2,3))
        mean = torch.sqrt(mean).to("cpu")
        
        self.values.append(mean)
        return mean
    
class LOG_RMSE_solidangle(PerformanceMetric):
    def __init__(self) -> None:
        super().__init__()
        
        self.values: list[torch.Tensor] = []
        self.name = self.__class__.__name__
        self.is_weight_initialized = False
        self.weight: torch.Tensor
        
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, i) -> torch.Tensor:
        self.reset_values(i)
        if not self.is_weight_initialized:
            self.is_weight_initialized = True
            self.weight = get_solid_angle_weights(y_true)
        device = y_true.device
        self.weight = self.weight.to(device)   
        mean = torch.mean(self.weight*torch.pow((torch.log(y_true+1) - torch.log(y_pred+1)),2), dim=(1,2,3))
        
        mean = torch.sqrt(mean).to("cpu")
        
        self.values.append(mean)
        return mean

# Not a real performance metric, but a helper function to calculate the avg luminance for the ground truth
class AVG_Luminance_GT(PerformanceMetric):
    def __init__(self) -> None:
        super().__init__()
        
        self.values: list[torch.Tensor] = []
        self.name = self.__class__.__name__
    
    def forward(self, _: torch.Tensor, y_true: torch.Tensor, i) -> torch.Tensor:
        self.reset_values(i)
        mean = torch.mean(y_true, dim=(1,2,3)).to("cpu")
        self.values.append(mean)
        return mean
               
if __name__ == "__main__":
    all_metrics = [MSE_solidangle(), MSE(), RMSE(), MAE(), MAE_solidangle(), RMSE_solidangle()]
    
    x_orig = torch.ones(2,3, 256, 512)
    y_orig_rand = torch.ones(2, 3, 256, 512) + 0.1
    scaling = 200
    scaled_y = y_orig_rand * scaling
    scaled_x = x_orig * scaling
    for metric in all_metrics:
        try:
            value = metric(x_orig, y_orig_rand)
            value_y_scaled = metric(x_orig, scaled_y)
            value_both_scaled = metric(scaled_x, scaled_y)
            print(f"{metric.name}: {value}, y*{scaling}: {value_y_scaled},y,x * {scaling}: {value_both_scaled}")
        except Exception as e:
            print(f"Error in {metric.name}: {e}")
        
        