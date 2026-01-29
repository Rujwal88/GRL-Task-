import logging
import os
import psutil
import functools
import time
import threading
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

class PerformanceMonitor(threading.Thread):
    """
    Background thread to monitor and log CPU/RAM usage in real-time.
    """
    def __init__(self, interval=0.5, func_name="Unknown"):
        super().__init__()
        self.interval = interval
        self.func_name = func_name
        self.running = True
        self.process = psutil.Process(os.getpid())

    def run(self):
        while self.running:
            try:
                cpu_p = self.process.cpu_percent(interval=None) # Non-blocking
                mem_info = self.process.memory_info()
                mem_mb = mem_info.rss / (1024 * 1024)
                
                # Only log if there's significant activity or at intervals
                # For this demo, we log every sample to prove it works
                logger.info(
                    f"   Create a Monitor 📈 [MONITOR] {self.func_name} running... "
                    f"CPU: {cpu_p:.1f}% | RAM: {mem_mb:.2f}MB"
                )
                time.sleep(self.interval)
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                break

    def stop(self):
        self.running = False


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
        mem_before = process.memory_info().rss / (1024 * 1024)  # MB
        
        start_time = time.perf_counter()
        
        # Start Real-Time Monitor
        monitor = PerformanceMonitor(interval=0.5, func_name=func.__name__)
        monitor.start()

        try:
            result = func(*args, **kwargs)
            
            # Format return value for logging (truncate if too long)
            ret_val_str = str(result)
            if len(ret_val_str) > 200:
                ret_val_str = ret_val_str[:200] + "... (truncated)"
            
            # Log Exit & Performance
            end_time = time.perf_counter()
            mem_after = process.memory_info().rss / (1024 * 1024)
            duration = end_time - start_time
            
            logger.info(
                f"Exiting {func.__name__}\n"
                f"   ⏱️  Duration: {duration:.4f}s\n"
                f"   💾 Memory: {mem_before:.2f}MB -> {mem_after:.2f}MB (Delta: {mem_after - mem_before:+.2f}MB)\n"
                f"   ⚙️  CPU Usage (Process): {process.cpu_percent(interval=None):.1f}%\n"
                f"   🔙 Return: {_get_obj_info(result)} :: {ret_val_str}"
            )
            
            return result
        
        except Exception as e:
            logger.error(f"CRITICAL ERROR in {func.__name__}: {str(e)}", exc_info=True)
            raise
        finally:
            # Ensure monitor stops even if function crashes
            monitor.stop()
            monitor.join(timeout=1.0)
            
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
