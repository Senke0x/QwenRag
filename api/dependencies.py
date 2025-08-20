"""
FastAPI依赖注入
"""
from functools import lru_cache
from typing import Optional

from clients.qwen_client import QwenClient
from processors.face_processor import FaceProcessor, FaceProcessorConfig
from pipelines.indexing_pipeline import IndexingPipeline
from pipelines.retrieval_pipeline import RetrievalPipeline
from api.services.search_service import SearchService
from api.services.face_service import FaceService
from api.services.indexing_service import IndexingService
from api.config import api_config

# 全局组件实例缓存
_qwen_client: Optional[QwenClient] = None
_face_processor: Optional[FaceProcessor] = None
_indexing_pipeline: Optional[IndexingPipeline] = None
_retrieval_pipeline: Optional[RetrievalPipeline] = None
_search_service: Optional[SearchService] = None
_face_service: Optional[FaceService] = None
_indexing_service: Optional[IndexingService] = None

@lru_cache()
def get_qwen_client() -> QwenClient:
    """获取Qwen客户端实例（单例）"""
    global _qwen_client
    if _qwen_client is None:
        _qwen_client = QwenClient()
    return _qwen_client

@lru_cache()
def get_face_processor() -> FaceProcessor:
    """获取人脸处理器实例（单例）"""
    global _face_processor
    if _face_processor is None:
        # 使用API配置创建人脸处理器配置
        face_config = FaceProcessorConfig(
            similarity_threshold=api_config.face_similarity_threshold,
            max_faces_per_image=api_config.max_faces_per_request,
            embedding_dimension=1536
        )
        _face_processor = FaceProcessor(get_qwen_client(), face_config)
    return _face_processor

@lru_cache()
def get_indexing_pipeline() -> IndexingPipeline:
    """获取索引构建管道实例（单例）"""
    global _indexing_pipeline
    if _indexing_pipeline is None:
        _indexing_pipeline = IndexingPipeline(
            batch_size=api_config.index_batch_size,
            max_workers=api_config.max_concurrent_indexing
        )
    return _indexing_pipeline

@lru_cache()
def get_retrieval_pipeline() -> RetrievalPipeline:
    """获取检索管道实例（单例）"""
    global _retrieval_pipeline
    if _retrieval_pipeline is None:
        _retrieval_pipeline = RetrievalPipeline()
    return _retrieval_pipeline

def get_search_service() -> SearchService:
    """获取搜索服务实例"""
    global _search_service
    if _search_service is None:
        _search_service = SearchService(
            retrieval_pipeline=get_retrieval_pipeline(),
            face_processor=get_face_processor()
        )
    return _search_service

def get_face_service() -> FaceService:
    """获取人脸服务实例"""
    global _face_service
    if _face_service is None:
        _face_service = FaceService(
            face_processor=get_face_processor()
        )
    return _face_service

def get_indexing_service() -> IndexingService:
    """获取索引服务实例"""
    global _indexing_service
    if _indexing_service is None:
        _indexing_service = IndexingService(
            indexing_pipeline=get_indexing_pipeline(),
            face_processor=get_face_processor()
        )
    return _indexing_service

# 清理函数
def cleanup_dependencies():
    """清理所有依赖组件"""
    global _qwen_client, _face_processor, _indexing_pipeline
    global _retrieval_pipeline, _search_service, _face_service, _indexing_service
    
    # 这里可以添加具体的清理逻辑，比如关闭数据库连接、保存状态等
    if _face_processor:
        try:
            _face_processor.save_face_index()
        except Exception:
            pass  # 忽略保存错误
    
    # 重置所有实例
    _qwen_client = None
    _face_processor = None
    _indexing_pipeline = None
    _retrieval_pipeline = None
    _search_service = None
    _face_service = None
    _indexing_service = None
    
    # 清除缓存
    get_qwen_client.cache_clear()
    get_face_processor.cache_clear()
    get_indexing_pipeline.cache_clear()
    get_retrieval_pipeline.cache_clear()