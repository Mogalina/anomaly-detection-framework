import sys
from pathlib import Path
from loguru import logger
from typing import Optional


def setup_logger(
    level: str = "INFO",
    log_file: Optional[str] = None,
    rotation: str = "100 MB",
    retention: str = "30 days",
    format_type: str = "json"
) -> None:
    """
    Configure the logger with specified settings.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file, if None only logs to stderr
        rotation: Log rotation size
        retention: Log retention period
        format_type: Format type ('json' or 'text')
    """
    # Remove default handler
    logger.remove()
    
    # Define format
    if format_type == "json":
        log_format = (
            "{{"
            '"timestamp": "{time:YYYY-MM-DD HH:mm:ss.SSS}", '
            '"level": "{level}", '
            '"module": "{module}", '
            '"function": "{function}", '
            '"line": {line}, '
            '"message": "{message}"'
            "}}"
        )
    else:
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
    
    # Add stderr handler
    logger.add(
        sys.stderr,
        format=log_format,
        level=level,
        colorize=format_type == "text"
    )
    
    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_file,
            format=log_format,
            level=level,
            rotation=rotation,
            retention=retention,
            compression="zip"
        )
    
    logger.info(f"Logger initialized with level={level}, format={format_type}")


def get_logger(name: str):
    """
    Get a logger instance with the given name.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Logger instance
    """
    return logger.bind(logger_name=name)
