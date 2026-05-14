import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from copy import deepcopy


class Config:
    """Configuration manager for the anomaly detection framework."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary
        """
        self._config = deepcopy(config_dict)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'edge.model.hidden_size')
            default: Default value if key not found
        
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'edge.model.hidden_size')
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Configuration dictionary
        """
        return deepcopy(self._config)
    
    def __getitem__(self, key: str) -> Any:
        """Allow dict-style access."""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dict-style setting."""
        self.set(key, value)


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses default path.
    
    Returns:
        Config object
    """
    if config_path is None:
        # Try to find config in standard locations
        possible_paths = [
            Path("/etc/adf/config.yaml"),
            Path("config/config.yaml"),
            Path("../config/config.yaml"),
        ]
        
        for path in possible_paths:
            if path.exists():
                config_path = str(path)
                break
        
        if config_path is None:
            raise FileNotFoundError(
                "Could not find config.yaml. Please specify config_path."
            )
    
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return Config(config_dict)


# Global configuration instance
_config_instance: Optional[Config] = None


def get_config() -> Config:
    """
    Get the global configuration instance.
    
    Returns:
        Global Config object
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = load_config()
    return _config_instance


def init_config(config_path: Optional[str] = None) -> Config:
    """
    Initialize global configuration.
    
    Args:
        config_path: Path to config file
    
    Returns:
        Initialized Config object
    """
    global _config_instance
    _config_instance = load_config(config_path)
    return _config_instance
