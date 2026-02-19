import torch
import pickle
from pathlib import Path
from typing import Any, Dict, Optional
import json


def save_model(
    model: torch.nn.Module,
    path: str,
    metadata: Optional[Dict[str, Any]] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None
) -> None:
    """
    Save PyTorch model with metadata.
    
    Args:
        model: PyTorch model to save
        path: Path to save model
        metadata: Optional metadata dictionary
        optimizer: Optional optimizer state
        epoch: Optional epoch number
    """
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
    }
    
    if metadata:
        checkpoint['metadata'] = metadata
    
    if optimizer:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    
    torch.save(checkpoint, save_path)


def load_model(
    model: torch.nn.Module,
    path: str,
    device: str = 'cpu',
    load_optimizer: bool = False,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> Dict[str, Any]:
    """
    Load PyTorch model from checkpoint.
    
    Args:
        model: PyTorch model instance to load weights into
        path: Path to model checkpoint
        device: Device to load model on
        load_optimizer: Whether to load optimizer state
        optimizer: Optimizer instance to load state into
    
    Returns:
        Dictionary containing metadata, epoch, etc.
    """
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    result = {
        'model_class': checkpoint.get('model_class'),
        'metadata': checkpoint.get('metadata', {}),
        'epoch': checkpoint.get('epoch'),
    }
    
    if load_optimizer and optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        result['optimizer_loaded'] = True
    
    return result


def save_object(obj: Any, path: str) -> None:
    """
    Save Python object using pickle.
    
    Args:
        obj: Object to save
        path: Path to save object
    """
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'wb') as f:
        pickle.dump(obj, f)


def load_object(path: str) -> Any:
    """
    Load Python object from pickle file.
    
    Args:
        path: Path to pickle file
    
    Returns:
        Loaded object
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_json(data: Dict[str, Any], path: str, indent: int = 2) -> None:
    """
    Save dictionary as JSON file.
    
    Args:
        data: Dictionary to save
        path: Path to save JSON
        indent: JSON indentation
    """
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=indent)


def load_json(path: str) -> Dict[str, Any]:
    """
    Load JSON file as dictionary.
    
    Args:
        path: Path to JSON file
    
    Returns:
        Loaded dictionary
    """
    with open(path, 'r') as f:
        return json.load(f)
