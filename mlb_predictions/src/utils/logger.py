"""
Logging Module

This module provides logging functionality for the application, including:
- Log file management
- Error tracking
- Performance monitoring
"""

import logging
import os
from datetime import datetime
from pathlib import Path

class Logger:
    def __init__(self, name, log_dir="logs"):
        """
        Initialize logger.
        
        Args:
            name (str): Logger name
            log_dir (str): Directory for log files
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Create file handler
        log_file = os.path.join(
            log_dir,
            f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def info(self, message):
        """Log info message."""
        self.logger.info(message)
    
    def error(self, message):
        """Log error message."""
        self.logger.error(message)
    
    def warning(self, message):
        """Log warning message."""
        self.logger.warning(message)
    
    def debug(self, message):
        """Log debug message."""
        self.logger.debug(message)
    
    def critical(self, message):
        """Log critical message."""
        self.logger.critical(message)

# Create loggers for different components
data_logger = Logger("data_collector")
model_logger = Logger("prediction_model")
api_logger = Logger("api")
feature_logger = Logger("feature_engineering")

def log_error(logger, error, context=None):
    """
    Log error with context information.
    
    Args:
        logger (Logger): Logger instance
        error (Exception): Error to log
        context (dict, optional): Additional context information
    """
    error_message = f"Error: {str(error)}"
    if context:
        error_message += f"\nContext: {context}"
    logger.error(error_message)

def log_performance(logger, operation, duration, details=None):
    """
    Log performance metrics.
    
    Args:
        logger (Logger): Logger instance
        operation (str): Operation name
        duration (float): Operation duration in seconds
        details (dict, optional): Additional performance details
    """
    message = f"Performance - {operation}: {duration:.2f} seconds"
    if details:
        message += f"\nDetails: {details}"
    logger.info(message)

def cleanup_old_logs(log_dir="logs", days_to_keep=30):
    """
    Remove log files older than specified days.
    
    Args:
        log_dir (str): Directory containing log files
        days_to_keep (int): Number of days to keep logs
    """
    cutoff_date = datetime.now() - timedelta(days=days_to_keep)
    
    for log_file in Path(log_dir).glob("*.log"):
        file_date = datetime.fromtimestamp(log_file.stat().st_mtime)
        if file_date < cutoff_date:
            try:
                log_file.unlink()
            except Exception as e:
                print(f"Error deleting old log file {log_file}: {str(e)}")

if __name__ == "__main__":
    # Example usage
    logger = Logger("test")
    
    logger.info("This is an info message")
    logger.error("This is an error message")
    logger.warning("This is a warning message")
    
    try:
        raise ValueError("Test error")
    except Exception as e:
        log_error(logger, e, {"test": "context"})
    
    log_performance(logger, "test_operation", 1.5, {"iterations": 100}) 