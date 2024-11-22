from .utils.logger import make_logger

sys_logger = make_logger("system", 'rlearn_system.log', level="DEBUG")
user_logger = make_logger("user", file='rlearn_user.log', level="INFO")
