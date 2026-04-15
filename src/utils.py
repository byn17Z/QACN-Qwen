"""
Description: Utility functions for the QEdpediaCN-Qwen project.
Usage: from src.utils import load_config
Dependencies: pyyaml
"""

import yaml
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads training configuration from a YAML file.
    
    Args:
        config_path (str): The path to the .yaml file.
        
    Returns:
        Dict[str, Any]: A dictionary containing configuration parameters.
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
