import logging
import os
import psutil
import functools
import time
import threading
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

    # Formatter: Timestamp - Level - Message
    # The message itself will often contain the metrics
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S,%f'
    )
    # Note: %f gives microseconds (e.g., 512345). We want 'mmm' (milliseconds).
    # Standard python logging doesn't have %3f. We'll stick to default %f and maybe trim it or just accept it.
    # Actually, let's fix the datefmt to be close. The default comma separator comes from the formatter.
    # The user asked for "2025-08-27 12:14:32,646".
    # logging default uses comma for msecs.
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )

    # File Handler
    file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8', mode='a') # Append mode to prevent truncation by subprocesses
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console Handler
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

class background_monitoring(threading.Thread):
    """
    Background thread to monitor and log CPU/RAM usage continuously.
    """
    def __init__(self, interval=0.5):
        super().__init__()
        self.interval = interval
        self.running = True
        self.process = psutil.Process(os.getpid())
        self.daemon = True # Daemon thread ensuring it dies with main process
        # Prime CPU counter
        self.process.cpu_percent(interval=None)

    def run(self):
        while self.running:
            try:
                time.sleep(self.interval)
                if not self.running: break
                
                cpu_p = self.process.cpu_percent(interval=None)
                mem_info = self.process.memory_info()
                mem_mb = mem_info.rss / (1024 * 1024)
                
                # Log continuously as requested
                logger.info(f"Background Monitor | CPU: {cpu_p:.1f}% | Memory: {mem_mb:.2f} MB")
                
            except Exception:
                break

    def stop(self):
        self.running = False


def log_performance(func):
    """
    Decorator to log:
    - Function Entry
    - Execution Duration
    - Continuous background monitoring during execution
    - Final summary with CPU/Memory
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        # Prime CPU
        process.cpu_percent(interval=None)
        
        func_name = func.__name__
        logger.info(f"Starting execution of: {func_name}")
        
        start_time = time.perf_counter()
        
        # Start Continuous Monitor
        # We start a NEW monitor for each decorated function to get granular logs, 
        # or we could rely on a global one. A per-function monitor allows us to see metrics *during* that function.
        monitor = background_monitoring(interval=1.0) # Log every 1 second
        monitor.start()

        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            logger.error(f"Error in {func_name}: {e}")
            raise
        finally:
            monitor.stop()
            monitor.join(timeout=1.0)
            
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            
            # Final Snapshot
            cpu_usage = process.cpu_percent(interval=None)
            mem_usage_mb = process.memory_info().rss / (1024 * 1024)
            
            # Specific format requested:
            # "... - INFO - Frame processed in 595.21 ms ..., CPU: 13.3%, Memory: 202.64 MB"
            # Adapting "Frame processed" to "Task executed" or "{func_name} executed"
            logger.info(
                f"{func_name} executed in {duration_ms:.2f} ms, CPU: {cpu_usage:.1f}%, Memory: {mem_usage_mb:.2f} MB"
            )

    return wrapper
