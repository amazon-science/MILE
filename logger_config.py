import logging
from pathlib import Path
import datetime

def setup_logging(log_dir='logs', log_level=logging.INFO):
    # Create logs directory if it doesn't exist
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    
    # Generate a unique filename for this run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"MILE_{timestamp}.log"
    
    file_handler = logging.FileHandler(log_dir / log_file)
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Logging to file: {log_dir / log_file}")

    return logger

# Create a logger object only if it hasn't been created yet
if not logging.getLogger().handlers:
    logger = setup_logging()
else:
    logger = logging.getLogger(__name__)