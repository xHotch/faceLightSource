import json
import os


def write_metrics(metrics: dict[str, float], folder: str) -> None:
    total_metrics:dict[str,dict[int, float]] = {}
    
    with open(os.path.join(folder, "overview_metrics.txt"), "w") as f:
        for _metric, _value in metrics.items():
            f.write(f"{_metric}: {_value}\n")
    sorted_total_metrics = {}
    for _metric, values in total_metrics.items():
        sorted_total_metrics[str(_metric)] = sorted(values.items(), key=lambda x: x[1])
    
    with open(os.path.join(folder, "total_metrics_sorted.json"), "w") as f:
        json.dump(sorted_total_metrics, f)