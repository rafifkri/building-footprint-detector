"""
Configuration loading and management utilities.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from omegaconf import OmegaConf, DictConfig


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config


def load_config_omega(config_path: str) -> DictConfig:
    """
    Load configuration using OmegaConf.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        OmegaConf DictConfig object
    """
    return OmegaConf.load(config_path)


def merge_configs(
    base_config: Dict[str, Any],
    override_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Merge two config dictionaries, with override taking precedence.
    
    Args:
        base_config: Base configuration
        override_config: Override configuration
        
    Returns:
        Merged configuration
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save config file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def get_nested_value(
    config: Dict[str, Any],
    key_path: str,
    default: Any = None,
) -> Any:
    """
    Get nested value from config using dot notation.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated key path (e.g., "model.encoder")
        default: Default value if key not found
        
    Returns:
        Value at key path or default
    """
    keys = key_path.split(".")
    value = config
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value


def set_nested_value(
    config: Dict[str, Any],
    key_path: str,
    value: Any,
) -> Dict[str, Any]:
    """
    Set nested value in config using dot notation.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated key path
        value: Value to set
        
    Returns:
        Modified configuration
    """
    keys = key_path.split(".")
    current = config
    
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    current[keys[-1]] = value
    
    return config


class Config:
    """Configuration class with attribute access."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert back to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get attribute with default."""
        return getattr(self, key, default)
    
    def __repr__(self) -> str:
        return f"Config({self.to_dict()})"
