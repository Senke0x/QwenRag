"""
配置模块
"""
import os

# 导入根级config模块的内容
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from config import Config, LogConfig, QwenVLConfig, RetryConfig, config
except ImportError:
    # 如果无法导入根级config，尝试相对路径导入
    try:
        import importlib.util

        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "config.py"
        )
        spec = importlib.util.spec_from_file_location("config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        QwenVLConfig = config_module.QwenVLConfig
        RetryConfig = config_module.RetryConfig
        LogConfig = config_module.LogConfig
        Config = config_module.Config
        config = config_module.config
    except Exception:
        # 如果都失败了，提供默认值
        QwenVLConfig = None
        RetryConfig = None
        LogConfig = None
        Config = None
        config = None

from .embedding_config import EmbeddingProcessorConfig, default_embedding_config


# 临时定义ImageProcessorConfig以避免导入错误
class ImageProcessorConfig:
    def __init__(self):
        self.supported_formats = [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"]
        self.max_image_size = 10 * 1024 * 1024  # 10MB


# 创建默认实例
default_image_processor_config = ImageProcessorConfig()

# 为了兼容性，也设置为ImageProcessorConfig
ImageProcessorConfig = default_image_processor_config

__all__ = [
    "QwenVLConfig",
    "RetryConfig",
    "LogConfig",
    "Config",
    "config",
    "EmbeddingProcessorConfig",
    "default_embedding_config",
    "ImageProcessorConfig",
    "default_image_processor_config",
]
