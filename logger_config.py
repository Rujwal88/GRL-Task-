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
    # Ensure stdout handles unicode (emojis) on Windows
    if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except:
            pass
            
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

# Create the logger instance
logger = setup_logger()

def _get_obj_info(obj):
    """
    Helper to get type and size/shape/len details of an object.
    Current support: standard types, numpy, torch, pandas.
    """
    type_name = type(obj).__name__
    details = ""
    
    try:
        if hasattr(obj, '__len__'):
            details = f"len={len(obj)}"
        elif hasattr(obj, 'shape'): # Numpy / Torch
            details = f"shape={obj.shape}"
        elif hasattr(obj, 'size'): # Numpy
            details = f"size={obj.size}"
            
        # Specific overrides for common libs if available in context
        # (We rely on duck typing/attributes to avoid strict dependencies here)
        
    except:
        pass # Fallback if inspection fails
        
    return f"<{type_name}{f' ({details})' if details else ''}>"

def log_performance(func):
    """
    Decorator to log:
    - Function Entry (with Args/Kwargs details)
    - Resource Usage (CPU %, Memory RSS)
    - Execution Time
    - Function Exit (with Return info)
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        
        # --- PRE-EXECUTION METRICS ---
        # Note: cpu_percent(interval=None) returns 0.0 on first call or immediate since last call.
        # To get meaningful value for *this* function, we might need a small interval, 
        # but blocking is bad. We'll rely on process-wide stats relative to system.
        # Calling it once here to 'reset' the counter for the next call if we wanted interval,
        # but for instant usage, we just log what we have.
        
        # Better approach for CPU: Measure CPU times before and after? 
        # Or just log system percent. Let's stick to process memory mainly and system cpu.
        cpu_before = process.cpu_percent(interval=None) 
        mem_before_bytes = process.memory_info().rss
        mem_before_mb = mem_before_bytes / (1024 * 1024)
        
        # Analyze Arguments
        args_details = [_get_obj_info(a) for a in args]
        kwargs_details = {k: _get_obj_info(v) for k, v in kwargs.items()}
        
        logger.debug(
            f"Entering {func.__name__} | "
            f"Args: {args_details} | Kwargs: {kwargs_details}"
        )
        
        start_time = time.perf_counter()
        
        try:
            result = func(*args, **kwargs)
            
            # --- POST-EXECUTION METRICS ---
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            mem_after_bytes = process.memory_info().rss
            mem_after_mb = mem_after_bytes / (1024 * 1024)
            mem_delta_mb = mem_after_mb - mem_before_mb
            
            # CPU usage during this interval (approximate)
            cpu_after = process.cpu_percent(interval=None)
            
            # Return value analysis
            ret_info = _get_obj_info(result)
            
            # Truncate string rep for logging if it's a simple type
            ret_str_preview = str(result)
            if len(ret_str_preview) > 100:
                ret_str_preview = ret_str_preview[:100] + "..."
            
            logger.info(
                f"Exiting {func.__name__}\n"
                f"   ⏱️  Duration: {duration:.4f}s\n"
                f"   💾 Memory: {mem_before_mb:.2f}MB -> {mem_after_mb:.2f}MB (Delta: {mem_delta_mb:+.2f}MB)\n"
                f"   ⚙️  CPU Usage (Process): {cpu_after:.1f}%\n"
                f"   🔙 Return: {ret_info} :: {ret_str_preview}"
            )
            
            return result
        
        except Exception as e:
            logger.error(f"CRITICAL ERROR in {func.__name__}: {str(e)}", exc_info=True)
            raise
            
    return wrapper

if __name__ == "__main__":
    # Test the logger and decorator
    logger.info("Testing logger_config.py directly...")
    
    @log_performance
    def test_function(n, data):
        logger.info(f"Inside test function with n={n}")
        time.sleep(0.1)
        return [x * 2 for x in data]

    test_function(5, data=[1, 2, 3, 4, 5])
    logger.info("Test complete.")
