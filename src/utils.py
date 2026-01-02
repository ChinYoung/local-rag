"""
工具函数模块
"""

import logging
import time
from functools import wraps
import requests
import ollama

logger = logging.getLogger(__name__)


def retry_on_failure(max_retries=3, delay=1):
    """重试装饰器，处理网络问题"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (
                    requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout,
                    ollama.ResponseError,
                ) as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = delay * (2**attempt)  # 指数退避
                        logger.warning(
                            f"连接失败，{wait_time}秒后重试 (尝试 {attempt + 1}/{max_retries}): {e}"
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(f"所有重试失败: {e}")
                        raise ConnectionError(f"无法连接到远程Ollama服务器: {e}")
                except Exception as e:
                    logger.error(f"未知错误: {e}")
                    raise
            raise BaseException("重试机制出现未知错误") from last_exception

        return wrapper

    return decorator
