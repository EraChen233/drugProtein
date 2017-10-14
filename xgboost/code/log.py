# coding:utf-8
import logging


def get_logger(name, log_file, stream=False):
    """
    生成一个logger对象，用于写日志
    Args:
        name: logger的名称
        log_file: 日志文件路径
    Returns:
        返回一个logger对象
    """
    format = "%(levelname)8s: %(asctime)s: %(filename)s:%(lineno)d %(message)s"
    datefmt = "%m-%d %H:%M:%S"

    # 输出到屏幕
    if stream == True:

        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        formatter = logging.Formatter(format, datefmt)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    # 输出到文件
    else:
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(format, datefmt)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger
