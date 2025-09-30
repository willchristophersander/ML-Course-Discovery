import os
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_project_root() -> Path:
    """Return the repository root as a Path object."""
    return _PROJECT_ROOT

def get_data_path(relative_path: str) -> str:
    """
    Get the correct absolute path for data files, whether running from root or src/scripts/
    
    Args:
        relative_path: Relative path from data directory (e.g., 'world_universities_and_domains.json')
    
    Returns:
        Absolute path to the data file
    """
    # Get the directory where the calling script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try different possible locations
    possible_paths = [
        # Running from src/scripts/
        os.path.join(script_dir, '..', 'data', relative_path),
        # Running from root directory
        os.path.join(script_dir, '..', 'src', 'data', relative_path),
        # Running from src/
        os.path.join(script_dir, 'data', relative_path),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # If none found, return the first path as default
    return possible_paths[0]


def get_models_path(filename: str) -> str:
    """Resolve a model artifact inside the central models directory."""
    candidate = get_project_root() / "models" / filename
    return str(candidate)
