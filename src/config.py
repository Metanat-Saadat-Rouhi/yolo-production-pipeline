import yaml
import os
from pydantic import BaseModel

class ModelConfig(BaseModel):
    weights: str
    conf_threshold: float
    iou_threshold: float
    device: str

class PipelineConfig(BaseModel):
    source: str
    enable_tracking: bool
    tracker_type: str
    save_output: bool
    show_window: bool

class AppConfig(BaseModel):
    model: ModelConfig
    pipeline: PipelineConfig

def load_config(config_path: str = "configs/default.yaml") -> AppConfig:
    # This line ensures we find the file relative to the project root, not the terminal
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    absolute_path = os.path.join(base_path, config_path)
    
    if not os.path.exists(absolute_path):
        raise FileNotFoundError(f"Config file not found at: {absolute_path}")
        
    with open(absolute_path, "r") as f:
        data = yaml.safe_load(f)
    return AppConfig(**data)