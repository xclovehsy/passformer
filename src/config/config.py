"""Configuration management for the Compiler project."""
import os
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


def convert_to_float(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    递归地遍历配置字典，将其中所有字符串形式的数字转换为 float 或 int 类型
    """
    for section, value in config_dict.items():
        if isinstance(value, dict):
            config_dict[section] = convert_to_float(value)
        elif isinstance(value, str):
            try:
                config_dict[section] = float(value)
            except ValueError:
                try:
                    config_dict[section] = int(value)
                except ValueError:
                    pass
    return config_dict


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and parse a YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return convert_to_float(cfg)


class Config:
    """Configuration class for training and experiments."""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
    SRC_ROOT = PROJECT_ROOT / "src"
    
    # Resource paths
    INST2VEC_RESOURCES = SRC_ROOT / "core" / "inst2vec" / "resources"
    INST2VEC_VOCAB_PATH = INST2VEC_RESOURCES / "dictionary.pickle"
    INST2VEC_EMBEDDING_PATH = INST2VEC_RESOURCES / "embeddings.pickle"
    
    def __init__(self, config_path: Optional[str] = None):
        self._cfg = {}
        if config_path:
            self._cfg = load_config(config_path)
    
    @property
    def model_id(self) -> str:
        return self._cfg.get("model", {}).get("model_id", "")
    
    @property
    def tokenizer_id(self) -> str:
        return self._cfg.get("model", {}).get("tokenizer_id", self.model_id)
    
    @property
    def data_dir(self) -> str:
        return self._cfg.get("data", {}).get("data_dir", "")
    
    @property
    def max_length(self) -> int:
        return self._cfg.get("data", {}).get("max_length", 512)
    
    @property
    def base_work_dir(self) -> str:
        return self._cfg.get("output", {}).get("base_work_dir", "./output")
    
    @property
    def mlm_probability(self) -> float:
        return self._cfg.get("mlm", {}).get("mlm_probability", 0.15)
    
    @property
    def training_args(self) -> Dict[str, Any]:
        args = self._cfg.get("training_args", {}).copy()
        if "learning_rate" in args:
            args["learning_rate"] = float(args["learning_rate"])
        return args
    
    def create_work_dir(self) -> str:
        """Create a timestamped work directory."""
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        work_dir = os.path.join(self.base_work_dir, time_str)
        os.makedirs(work_dir, exist_ok=True)
        return work_dir
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by dot-separated key."""
        keys = key.split(".")
        value = self._cfg
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value
    
    def __getitem__(self, key: str) -> Any:
        return self._cfg[key]
    
    def __contains__(self, key: str) -> bool:
        return key in self._cfg

