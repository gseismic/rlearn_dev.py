from .utils.logger import make_logger

sys_logger = make_logger("system", level="DEBUG")
user_logger = make_logger("user", level="INFO")