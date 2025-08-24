"""
QwenRag - Qwen VL图像检索系统
"""

__version__ = "0.1.0"
__author__ = "QwenRag Team"
__description__ = "基于Qwen VL的图像检索和向量存储系统"

from .config import config
from .processors.image_processor import ImageProcessor
from .schemas.data_models import ImageMetadata, ProcessingStatus
from .vector_store.faiss_store import FaissStore

__all__ = [
    "config",
    "ImageMetadata",
    "ProcessingStatus",
    "ImageProcessor",
    "FaissStore",
]
