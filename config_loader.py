import json
import os
from logger_config import logger

CONFIG_FILE = "config.json"

def load_config(config_path=CONFIG_FILE):
    """
    Load configuration from a JSON file.
    
    Args:
        config_path (str): Path to the config file.
        
    Returns:
        dict: Configuration dictionary.
    """
    if not os.path.exists(config_path):
        logger.warning(f"⚠️  Config file not found at {config_path}. Using defaults.")
        # Default fallback
        return {
            "tts_model": "xtts_v2",
            "audio_format": "wav",
            "sample_rate": 22050,
            "model_settings": {}
        }

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            logger.info(f"✅ Configuration loaded from {config_path}")
            logger.debug(f"   Config content: {config}")
            return config
    except Exception as e:
        logger.error(f"❌ Failed to load config: {e}")
        raise
