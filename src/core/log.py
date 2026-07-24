import logging
import os
import sys

from core.log_context import get_request_id

class RequestIdFilter(logging.Filter):
    
    def filter(self, record):
        if not hasattr(record, "request_id") or record.request_id == "-":
            record.request_id = get_request_id()
        return True
    

def get_logger(name):
    # create a logger
    logger = logging.getLogger(name)
    
    # disable propagation to avoid double logging
    logger.propagate = False
    
    # clear existing handlers if any
    if logger.hasHandlers():
        logger.handlers.clear()
        
    # set log level
    if os.environ.get("LOG_LEVEL", "INFO") == "DEBUG":
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
        
    # create console handler and set level to debug
    console_handler = logging.StreamHandler(sys.stdout)
    
    if os.environ.get("LOG_LEVEL", "INFO") == "DEBUG":
        console_handler.setLevel(logging.DEBUG)
    else:
        console_handler.setLevel(logging.INFO)
        
    # update formatter - add request_id
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - [%(filename)s:%(lineno)d] - [request_id:%(request_id)s] - %(message)s'
    )
    
    # add formatter to console handler
    console_handler.setFormatter(formatter)
    
    # add RequestIdFitler to inject request_id
    console_handler.addFilter(RequestIdFilter())
    
    # add console handler to logger
    logger.addHandler(console_handler)
    
    return logger
