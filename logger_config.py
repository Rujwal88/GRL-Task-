import logging
import os
import psutil
import functools
import time
from logging.handlers import RotatingFileHandler
import sys

# Ensure logs directory exists
LOG_DIR = 'logs'
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

LOG_FILE = os.path.join(LOG_DIR, 'voice_cloning.log')

def setup_logger(name="voice_cloning"):
    """
    Sets up a logger with DEBUG level and RotatingFileHandler.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Prevent adding handlers multiple times if function called repeatedly
    if logger.handlers:
        return logger

    # Formatter: Timestamp - Level - Module - Function - Message
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(module)s - %(funcName)s() - %(message)s'
    )

    # File Handler
    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=5*1024*1024, backupCount=5, encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

# Create the logger instance
logger = setup_logger()

def log_performance(func):
    """
    Decorator to log function entry, exit, execution time, and memory usage.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Log Entry
        logger.debug(f"Entering {func.__name__} with args: {args}, kwargs: {kwargs}")
        
        # Memory Before
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024)  # MB
        
        start_time = time.perf_counter()
        
        try:
            result = func(*args, **kwargs)
            
            # End Time & Memory After
            end_time = time.perf_counter()
            mem_after = process.memory_info().rss / (1024 * 1024)
            duration = end_time - start_time
            
            # Simplified Complexity Note
            # O(1) for constants, O(N) where N is input size usually
            
            # Format return value for logging (truncate if too long)
            ret_val_str = str(result)
            if len(ret_val_str) > 200:
                ret_val_str = ret_val_str[:200] + "... (truncated)"
            
            # Log Exit & Performance
            logger.info(
                f"Exiting {func.__name__} - "
                f"Duration: {duration:.4f}s - "
                f"Memory: {mem_before:.2f}MB -> {mem_after:.2f}MB "
                f"(Delta: {mem_after - mem_before:+.2f}MB) - "
                f"Return: {type(result).__name__} = {ret_val_str}"
            )
            
            return result
        
        except Exception as e:
            logger.error(f"CRITICAL ERROR in {func.__name__}: {str(e)}", exc_info=True)
            raise
            
    return wrapper
