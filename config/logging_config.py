"""
模块一：日志配置模块
为什么需要配置日志？因为日志可以帮助我们更好的去了解程序的运行情况，快速定位出现问题的地方

日志的级别
DEBUG INFO WARNING ERROR CRITICAL

写好了日志配置模块之后，在主程序中调用，在其他的任意模块中可以继承该全局日志

"""

import logging
import sys

from config.settings import settings # 获取实例化的管理器
from logging.handlers import RotatingFileHandler

def setup_logging():
    """
    配置全局日志的函数
    """
    # 日志的格式：时间 - 日志器名称 - 日志级别 - 日志内容
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    # 时间的格式
    date_format = "%Y-%m-%d %H:%M:%S"

    # 根日志器
    root_logger = logging.getLogger()
    # 设置日志的等级
    root_logger.setLevel(getattr(logging, settings.LOG_LEVEL))

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    # 设置日志的等级
    console_handler.setLevel(getattr(logging, settings.LOG_LEVEL))
    # 日志的格式
    console_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    # 添加到根日志器
    root_logger.addHandler(console_handler)

    # 文件处理器（带轮转的）
    # 设置日志的文件大小，当达到该大小的时候，会自动进行轮转
    file_handler = RotatingFileHandler(
        settings.LOG_FILE,                  # 日志文件保存带路径的文件的文件名
        maxBytes=10 * 1024 * 1024,          # 10MB
        backupCount=5,                      # 保留5个备份
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG) # 所有的日志信息都保存到日志文件中
    file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    root_logger.addHandler(file_handler) # 添加到根日志器

    # 减少第三方库的日志噪音（可选）
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)

    # 测试日志
    root_logger.info(f"日志系统初始化完成，级别是{settings.LOG_LEVEL}")
