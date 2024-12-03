from .utils.logger import make_logger

sys_logger = make_logger("system", file=None, level="DEBUG")
user_logger = make_logger("user", file=None, level="INFO")
