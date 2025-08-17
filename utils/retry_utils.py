"""
重试机制工具函数
"""
import time
import random
from typing import Callable, Any, Optional, Type, Tuple
from functools import wraps
import logging

from config import RetryConfig

logger = logging.getLogger(__name__)


class RetryableError(Exception):
    """可重试的错误"""
    pass


class NonRetryableError(Exception):
    """不可重试的错误"""
    pass


def calculate_delay(attempt: int, config: RetryConfig) -> float:
    """
    计算重试延迟时间（指数退避 + 抖动）
    
    Args:
        attempt: 重试次数（从1开始）
        config: 重试配置
    
    Returns:
        延迟时间（秒）
    """
    # 指数退避
    delay = config.base_delay * (config.exponential_base ** (attempt - 1))
    
    # 限制最大延迟
    delay = min(delay, config.max_delay)
    
    # 添加随机抖动（避免惊群效应）
    jitter = random.uniform(0.1, 0.3) * delay
    
    return delay + jitter


def is_retryable_error(error: Exception, config: RetryConfig) -> bool:
    """
    判断错误是否可重试
    
    Args:
        error: 异常对象
        config: 重试配置
    
    Returns:
        是否可重试
    """
    if isinstance(error, NonRetryableError):
        return False
    
    if isinstance(error, RetryableError):
        return True
    
    # 根据错误消息判断
    error_message = str(error).lower()
    
    for retryable_pattern in config.retryable_errors:
        if retryable_pattern.lower() in error_message:
            return True
    
    # HTTP状态码判断
    if hasattr(error, 'response') and hasattr(error.response, 'status_code'):
        status_code = error.response.status_code
        # 5xx错误和429（限流）可重试
        if status_code >= 500 or status_code == 429:
            return True
        # 4xx错误（除429外）不可重试
        if 400 <= status_code < 500:
            return False
    
    return False


def retry_with_backoff(
    config: Optional[RetryConfig] = None,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
):
    """
    重试装饰器，支持指数退避
    
    Args:
        config: 重试配置
        exceptions: 需要捕获的异常类型
    
    Returns:
        装饰器函数
    """
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(config.max_retries + 1):  # +1 for initial attempt
                try:
                    result = func(*args, **kwargs)
                    
                    # 如果不是第一次尝试，记录成功重试
                    if attempt > 0:
                        logger.info(
                            f"函数 {func.__name__} 在第 {attempt + 1} 次尝试时成功"
                        )
                    
                    return result
                
                except exceptions as e:
                    last_exception = e
                    
                    # 检查是否可重试
                    if not is_retryable_error(e, config):
                        logger.error(
                            f"函数 {func.__name__} 遇到不可重试错误: {e}"
                        )
                        raise
                    
                    # 如果已达到最大重试次数
                    if attempt >= config.max_retries:
                        logger.error(
                            f"函数 {func.__name__} 重试 {config.max_retries} 次后仍然失败: {e}"
                        )
                        raise
                    
                    # 计算延迟时间
                    delay = calculate_delay(attempt + 1, config)
                    
                    logger.warning(
                        f"函数 {func.__name__} 第 {attempt + 1} 次尝试失败: {e}, "
                        f"{delay:.2f}秒后重试"
                    )
                    
                    time.sleep(delay)
            
            # 理论上不会到达这里
            if last_exception:
                raise last_exception
        
        return wrapper
    return decorator


def retry_async_with_backoff(
    config: Optional[RetryConfig] = None,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
):
    """
    异步重试装饰器
    
    Args:
        config: 重试配置
        exceptions: 需要捕获的异常类型
    
    Returns:
        装饰器函数
    """
    import asyncio
    
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(config.max_retries + 1):
                try:
                    result = await func(*args, **kwargs)
                    
                    if attempt > 0:
                        logger.info(
                            f"异步函数 {func.__name__} 在第 {attempt + 1} 次尝试时成功"
                        )
                    
                    return result
                
                except exceptions as e:
                    last_exception = e
                    
                    if not is_retryable_error(e, config):
                        logger.error(
                            f"异步函数 {func.__name__} 遇到不可重试错误: {e}"
                        )
                        raise
                    
                    if attempt >= config.max_retries:
                        logger.error(
                            f"异步函数 {func.__name__} 重试 {config.max_retries} 次后仍然失败: {e}"
                        )
                        raise
                    
                    delay = calculate_delay(attempt + 1, config)
                    
                    logger.warning(
                        f"异步函数 {func.__name__} 第 {attempt + 1} 次尝试失败: {e}, "
                        f"{delay:.2f}秒后重试"
                    )
                    
                    await asyncio.sleep(delay)
            
            if last_exception:
                raise last_exception
        
        return wrapper
    return decorator