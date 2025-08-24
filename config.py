"""
配置文件，存放所有路径和参数
"""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class RetryConfig:
    """重试配置"""

    max_retries: int = 3
    base_delay: float = 1.0  # 基础延迟（秒）
    max_delay: float = 60.0  # 最大延迟（秒）
    exponential_base: int = 2  # 指数退避基数
    retryable_errors: List[str] = None

    def __post_init__(self):
        if self.retryable_errors is None:
            self.retryable_errors = [
                "rate_limit_exceeded",
                "service_unavailable",
                "timeout",
                "connection_error",
                "internal_error",
            ]


@dataclass
class QwenVLConfig:
    """Qwen VL API配置"""

    api_key: str = ""
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model: str = "qwen-vl-max-latest"
    timeout: int = 60
    max_tokens: int = 2048
    temperature: float = 0.1

    def __post_init__(self):
        if not self.api_key:
            self.api_key = os.getenv("DASHSCOPE_API_KEY", "")


@dataclass
class ImageProcessorConfig:
    """图片处理配置"""

    supported_formats: List[str] = None
    max_image_size: int = 20 * 1024 * 1024  # 20MB
    max_resolution: tuple = (4096, 4096)
    resize_quality: int = 85

    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = [".jpg", ".jpeg", ".png", ".webp"]


@dataclass
class Config:
    """主配置类"""

    qwen_vl: QwenVLConfig = None
    retry: RetryConfig = None
    image_processor: ImageProcessorConfig = None

    # 路径配置
    input_dir: str = ""
    output_dir: str = ""
    index_dir: str = ""

    # 日志配置
    log_level: str = "INFO"
    log_file: str = "qwen_rag.log"

    def __post_init__(self):
        if self.qwen_vl is None:
            self.qwen_vl = QwenVLConfig()
        if self.retry is None:
            self.retry = RetryConfig()
        if self.image_processor is None:
            self.image_processor = ImageProcessorConfig()


def load_config_from_yaml(config_path: str) -> Config:
    """从YAML文件加载配置"""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)

        # 创建配置对象
        config = Config()

        # 加载Qwen VL配置
        if "qwen_vl" in yaml_data:
            qwen_data = yaml_data["qwen_vl"]
            config.qwen_vl = QwenVLConfig(
                api_key=qwen_data.get("api_key", ""),
                base_url=qwen_data.get("base_url", config.qwen_vl.base_url),
                model=qwen_data.get("model", config.qwen_vl.model),
                timeout=qwen_data.get("timeout", config.qwen_vl.timeout),
                max_tokens=qwen_data.get("max_tokens", config.qwen_vl.max_tokens),
                temperature=qwen_data.get("temperature", config.qwen_vl.temperature),
            )

        # 加载重试配置
        if "retry" in yaml_data:
            retry_data = yaml_data["retry"]
            config.retry = RetryConfig(
                max_retries=retry_data.get("max_retries", config.retry.max_retries),
                base_delay=retry_data.get("base_delay", config.retry.base_delay),
                max_delay=retry_data.get("max_delay", config.retry.max_delay),
                exponential_base=retry_data.get(
                    "exponential_base", config.retry.exponential_base
                ),
                retryable_errors=retry_data.get(
                    "retryable_errors", config.retry.retryable_errors
                ),
            )

        # 加载图片处理配置
        if "image_processor" in yaml_data:
            img_data = yaml_data["image_processor"]
            config.image_processor = ImageProcessorConfig(
                supported_formats=img_data.get(
                    "supported_formats", config.image_processor.supported_formats
                ),
                max_image_size=img_data.get(
                    "max_image_size", config.image_processor.max_image_size
                ),
                max_resolution=tuple(
                    img_data.get(
                        "max_resolution", config.image_processor.max_resolution
                    )
                ),
                resize_quality=img_data.get(
                    "resize_quality", config.image_processor.resize_quality
                ),
            )

        # 加载路径配置
        if "paths" in yaml_data:
            paths_data = yaml_data["paths"]
            config.input_dir = paths_data.get("input_dir", "")
            config.output_dir = paths_data.get("output_dir", "")
            config.index_dir = paths_data.get("index_dir", "")

        # 加载日志配置
        if "logging" in yaml_data:
            log_data = yaml_data["logging"]
            config.log_level = log_data.get("level", config.log_level)
            config.log_file = log_data.get("file", config.log_file)

        return config

    except FileNotFoundError:
        print(f"配置文件不存在: {config_path}")
        return Config()
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        return Config()


def find_config_file() -> Optional[str]:
    """查找配置文件"""
    possible_paths = [
        "config.yaml",
        "config.yml",
        os.path.expanduser("~/.qwen_rag/config.yaml"),
        "/etc/qwen_rag/config.yaml",
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    return None


# 全局配置实例
_config_file = find_config_file()
if _config_file:
    config = load_config_from_yaml(_config_file)
else:
    config = Config()
