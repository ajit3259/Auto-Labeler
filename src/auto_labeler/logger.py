import logging
import sys
from typing import Union

def setup_logger(name: str = "auto_labeler", level: Union[int, str] = logging.INFO) -> logging.Logger:
    """
    Sets up a standardized logger for the library.
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
        
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # If the logger already has handlers, don't add more (prevents duplicate logs)
    if not logger.handlers:
        
        # Consol handler with simple format
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger

# Default library-wide logger
logger = setup_logger()
