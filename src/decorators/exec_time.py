from functools import wraps
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def exec_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        import time
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"Function {func.__name__} took {execution_time:.4f} seconds to execute")
        return result
    return wrapper