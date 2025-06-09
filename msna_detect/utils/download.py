import gdown
import os

# Registry of available pretrained models
PRETRAINED_MODELS = {
    "msna-v1": {
        "url": "https://drive.google.com/file/d/1aP6j-FdlhC21QxBYGF49_L31UChfL2V0/view?usp=sharing",
        "filename": "msna-v1.pt",
        "description": "MSNA burst detection model v1.0. Trained on simulated data."
    }
}


def download_file_from_google_drive(url: str, destination: str, quiet: bool = False) -> None:
    """Download a file from Google Drive using gdown."""

    # Extract file ID from Google Drive URL
    if "drive.google.com" in url:
        if "/file/d/" in url:
            file_id = url.split("/file/d/")[1].split("/")[0]
        elif "id=" in url:
            file_id = url.split("id=")[1].split("&")[0]
        else:
            raise ValueError(f"Cannot extract file ID from URL: {url}")
        
        # Use gdown's direct download URL format
        download_url = f"https://drive.google.com/uc?id={file_id}"
    else:
        download_url = url

    # Download with gdown
    gdown.download(download_url, destination, quiet = quiet)


def get_cache_dir() -> str:
    """Get the cache directory for storing downloaded models."""
    cache_dir = os.path.join(os.path.expanduser("~"), ".msna-detect", "models")
    os.makedirs(cache_dir, exist_ok = True)
    return cache_dir


def get_model_path(model_name: str) -> str:
    """Get the local path where a model should be stored."""
    if model_name not in PRETRAINED_MODELS:
        raise ValueError(
            f"Unknown pretrained model: {model_name}. Available "
            f"models: {list(PRETRAINED_MODELS.keys())}"
        )
    
    cache_dir = get_cache_dir()
    filename = PRETRAINED_MODELS[model_name]["filename"]

    return os.path.join(cache_dir, filename)


def download_pretrained_model(model_name: str, force_download: bool = False, quiet: bool = False) -> str:
    """
    Download a pretrained model if it doesn't exist locally.
    
    Args:
        model_name: Name of the pretrained model to download
        force_download: If True, download even if file exists locally
        quiet: If True, suppress output messages
        
    Returns:
        str: Path to the downloaded model file
    """
    if model_name not in PRETRAINED_MODELS:
        raise ValueError(f"Unknown pretrained model: {model_name}. Available models: {list(PRETRAINED_MODELS.keys())}")
    
    model_info = PRETRAINED_MODELS[model_name]
    model_path = get_model_path(model_name)
    
    # Check if model already exists
    if not force_download and os.path.exists(model_path):
        if not quiet:
            print(f"Model {model_name} already cached.")
        return model_path
    
    # Download the model
    if not quiet:
        print(f"Downloading {model_name}...")
    
    try:
        download_file_from_google_drive(model_info["url"], model_path, quiet = quiet)
        
        if not quiet:
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            print(f"Downloaded {model_name} ({size_mb:.1f}MB)")
            
        return model_path
        
    except Exception as e:
        if os.path.exists(model_path):
            os.remove(model_path)  # Clean up partial download

        raise RuntimeError(f"Failed to download model {model_name}: {e}")


def list_pretrained_models() -> None:
    """List all available pretrained models."""
    print("Available pretrained models:")
    for name, info in PRETRAINED_MODELS.items():
        model_path = get_model_path(name)
        status = "✓" if os.path.exists(model_path) else " "
        print(f"  {status} {name:<12} {info['description']}")


def clear_model_cache(quiet: bool = False) -> None:
    """Clear all downloaded models from cache."""
    cache_dir = get_cache_dir()
    if not os.path.exists(cache_dir):
        if not quiet:
            print("No model cache found.")
        return
    
    model_files = [f for f in os.listdir(cache_dir) if f.endswith('.pth')]
    if not model_files:
        if not quiet:
            print("No models found in cache.")
        return
    
    for model_file in model_files:
        file_path = os.path.join(cache_dir, model_file)
        os.remove(file_path)
        if not quiet:
            print(f"Deleted: {model_file}")
    
    if not quiet:
        print(f"Cleared {len(model_files)} model(s) from cache.")


