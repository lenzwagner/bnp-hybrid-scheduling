import logging
import sys
from datetime import datetime


# Define custom PRINT level (between INFO and WARNING)
PRINT_LEVEL = 25
logging.addLevelName(PRINT_LEVEL, 'PRINT')


def print_log(self, message, *args, **kwargs):
    """
    Custom logger method for PRINT level.
    Use: logger.print("message")
    """
    if self.isEnabledFor(PRINT_LEVEL):
        self._log(PRINT_LEVEL, message, args, **kwargs)


# Add print method to Logger class
logging.Logger.print = print_log


class LevelFilter(logging.Filter):
    """
    Filter that only allows logs of a specific level.
    
    This ensures that each log file contains only its designated level,
    without duplication across files.
    """
    def __init__(self, level):
        super().__init__()
        self.level = level
    
    def filter(self, record):
        return record.levelno == self.level


def setup_multi_level_logging(base_log_dir='logs', enable_console=True):
    """
    Configure logging with separate log files for each level.
    
    Creates the following structure:
    - logs/debug/bnp_TIMESTAMP.log    - Only DEBUG messages
    - logs/info/bnp_TIMESTAMP.log     - Only INFO messages
    - logs/warning/bnp_TIMESTAMP.log  - Only WARNING messages
    - logs/error/bnp_TIMESTAMP.log    - Only ERROR messages
    
    Console output: Only shows PRINT level (use logger.print("message"))
    
    Args:
        base_log_dir: Base directory for log files (default: 'logs')
        enable_console: If True, console will show PRINT level messages (default: True)
    
    Returns:
        root_logger: Configured root logger
    """
    import os
    
    # Create formatter for files
    file_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create simpler formatter for console (PRINT level only)
    console_formatter = logging.Formatter(
        fmt='%(message)s'  # Simple format for console
    )
    
    # Configure root logger (set to DEBUG to capture everything)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers
    root_logger.handlers = []
    
    # Console handler - ONLY for PRINT level
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(PRINT_LEVEL)
        console_handler.addFilter(LevelFilter(PRINT_LEVEL))  # Only PRINT level
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # Create timestamp for all log files
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Define log levels and their directories
    log_levels = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
    }
    
    # Create file handler for each level
    for level_name, level_num in log_levels.items():
        # Create directory
        log_dir = os.path.join(base_log_dir, level_name.lower())
        os.makedirs(log_dir, exist_ok=True)
        
        # Create log file
        log_file = os.path.join(log_dir, f'bnp_{timestamp}.log')
        
        # Create file handler
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(level_num)
        
        # Add level filter to ensure only this level is written
        file_handler.addFilter(LevelFilter(level_num))
        
        # Set formatter
        file_handler.setFormatter(file_formatter)
        
        # Add to root logger
        root_logger.addHandler(file_handler)
    
    return root_logger


def setup_logging(log_level='INFO', log_to_file=True, log_dir='logs'):
    """
    Configure logging for Branch-and-Price solver (single-file mode).
    
    This is the simple version that writes all logs to a single file.
    Use setup_multi_level_logging() for separate files per level.

    Args:
        log_level: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
        log_to_file: If True, also write logs to file
        log_dir: Directory for log files
    """
    # Convert string to logging level
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')

    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers
    root_logger.handlers = []

    # Console handler (always)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_to_file:
        import os
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'bnp_{timestamp}.log')

        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger


def get_logger(name):
    """Get a logger for a specific module."""
    return logging.getLogger(name)