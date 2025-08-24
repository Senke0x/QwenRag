"""
EmbeddingProcessor配置类
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class EmbeddingProcessorConfig:
    """EmbeddingProcessor配置类"""

    # 向量维度设置
    embedding_dimension: Optional[int] = None  # None表示自动检测
    default_embedding_dimension: int = 1024  # 默认维度

    # 存储设置
    index_save_path: str = "data/faiss_index"
    auto_save: bool = True

    # 批量处理设置
    batch_save_frequency: int = 10  # 每处理多少个项目保存一次索引
    batch_api_size: int = 32  # API批量调用大小
    max_batch_size: int = 100  # 最大批量处理大小

    # 搜索设置
    default_top_k: int = 10  # 默认搜索返回数量
    similarity_threshold: float = 0.8  # 相似度阈值

    # 人脸处理设置
    face_padding_ratio: float = 0.2  # 人脸裁剪时的边界填充比例

    # 性能设置
    enable_parallel_processing: bool = True  # 是否启用并行处理
    max_workers: int = 4  # 并行处理最大工作线程数

    # 错误处理设置
    max_retries: int = 3  # 最大重试次数
    retry_delay: float = 1.0  # 重试延迟（秒）

    def validate(self) -> None:
        """验证配置参数"""
        if self.embedding_dimension is not None and self.embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be positive")

        if self.default_embedding_dimension <= 0:
            raise ValueError("default_embedding_dimension must be positive")

        if self.batch_save_frequency <= 0:
            raise ValueError("batch_save_frequency must be positive")

        if self.batch_api_size <= 0:
            raise ValueError("batch_api_size must be positive")

        if self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")

        if not (0.0 < self.similarity_threshold <= 1.0):
            raise ValueError("similarity_threshold must be between 0 and 1")

        if not (0.0 <= self.face_padding_ratio <= 1.0):
            raise ValueError("face_padding_ratio must be between 0 and 1")

        if self.max_workers <= 0:
            raise ValueError("max_workers must be positive")

        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")

        if self.retry_delay < 0:
            raise ValueError("retry_delay must be non-negative")


# 默认配置实例
default_embedding_config = EmbeddingProcessorConfig()
