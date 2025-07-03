import sys
from loguru import logger

def make_logger(name=None, 
                file=None, 
                format=None, 
                level="INFO"):
    format = format or "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <6}</level> | {message}"
    log = logger.bind(name=name) if name else logger
    log.remove()
    log.add(sys.stdout, format=format, level=level, colorize=True)
    if file is not None:
        log.add(file, format=format, level=level)
    return log