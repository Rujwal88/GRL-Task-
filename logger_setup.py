import logging
import os
import time
import psutil
import functools
from logging.handlers import RotatingFileHandler

def setup_logger(name="voice_cloning"):
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Format: Timestamp - Level - Module - Function - Message
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(module)s - %(funcName)s() - %(message)s'
    )

    # File Handler (Rotates at 5MB, keeps 5 backup files)
    file_handler = RotatingFileHandler(
        'logs/voice_cloning.log', maxBytes=5*1024*1024, backupCount=5
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console Handler (To see INFO in your terminal)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

logger = setup_logger()

def log_performance(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Memory before
        mem_before = psutil.Process().memory_info().rss / (1024 * 1024)
        start_time = time.perf_counter()
        
        logger.info(f"START - Parameters: {args}, {kwargs}")
        
        try:
            result = func(*args, **kwargs)
            
            end_time = time.perf_counter()
            mem_after = psutil.Process().memory_info().rss / (1024 * 1024)
            
            duration = end_time - start_time
            logger.info(f"END - Duration: {duration:.3f}s, Memory Delta: {mem_after - mem_before:.2f} MB")
            return result
        except Exception as e:
            logger.error(f"EXCEPTION in {func.__name__}: {str(e)}", exc_info=True)
            raise
    return wrapper
