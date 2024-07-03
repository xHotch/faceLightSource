import json
import os
from lightestimation.evaluation.accuracy import PerformanceMetric

class ConfigLoader():
    @classmethod
    def _load_config(cls, file_path:str|None = None):
        config_path = ('config.json') if file_path is None else file_path
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        class_vars = vars(cls)
        for key, value in config.items():
            if key in class_vars:
                setattr(cls, key, value)

class Config(ConfigLoader):
    
    WANDB_RUN_DIR:str = ""
    WANDB_PROJECT_PATH: str = ""
    TEST_LIGHTPROBE_PATH:str  = ""
    TRAIN_LIGHTPROBE_PATH:str  = ""
    TEST_LIGHTPROBE_PATH_UNROTATED:str  = ""
    TRAIN_LIGHTPROBE_PATH_UNROTATED:str  = ""
    MODEL_OUTPUT_DIR: str = ""
    DISABLE_WANDB: bool = False
    GPU_ID: int|list[int] = -1
    IMAGE_LOG_FREQUENCY: int = 100
    WATCH_MODEL: bool = False
    PERFORMANCE_METRICS: list[str] = []
    
    @classmethod
    def get_performance_metrics(cls) -> list[PerformanceMetric]:
        metrics = []
        import importlib
        module = importlib.import_module("lightestimation.evaluation.accuracy")
        
        for metric_name in cls.PERFORMANCE_METRICS:
            class_ = getattr(module, metric_name)
            if not issubclass(class_, PerformanceMetric) or class_ == PerformanceMetric:
                raise ValueError(f"Class {class_} is not a subclass of PerformanceMetric")
            metrics.append(class_())

        return metrics
        
Config._load_config("config.json")