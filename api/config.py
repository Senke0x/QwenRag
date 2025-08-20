"""
API配置模块
"""
from typing import List, Optional
try:
    from pydantic_settings import BaseSettings
except ImportError:
    # 兼容旧版本 pydantic
    from pydantic import BaseSettings
import os

class APIConfig(BaseSettings):
    """API配置类"""
    
    # 服务配置
    app_name: str = "QwenRag API"
    version: str = "1.0.0"
    debug: bool = False
    
    # 服务器配置
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    
    # 文档配置
    enable_docs: bool = True
    
    # CORS配置
    allowed_origins: List[str] = ["*"]
    allowed_methods: List[str] = ["*"]
    allowed_headers: List[str] = ["*"]
    
    # 受信任的主机
    trusted_hosts: Optional[List[str]] = None
    
    # 上传文件配置
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_image_types: List[str] = [
        "image/jpeg", "image/jpg", "image/png", 
        "image/bmp", "image/gif", "image/webp"
    ]
    upload_dir: str = "temp_uploads"
    
    # 搜索配置
    default_search_limit: int = 10
    max_search_limit: int = 100
    default_similarity_threshold: float = 0.5
    
    # 人脸识别配置
    face_similarity_threshold: float = 0.8
    max_faces_per_request: int = 20
    
    # 索引配置
    index_batch_size: int = 10
    max_concurrent_indexing: int = 4
    
    class Config:
        env_prefix = "QWEN_RAG_"
        env_file = ".env"

# 全局配置实例
api_config = APIConfig()