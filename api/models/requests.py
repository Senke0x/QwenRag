"""
API请求数据模型
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from enum import Enum

class SearchMode(str, Enum):
    """搜索模式枚举"""
    TEXT = "text"
    IMAGE = "image"
    FACE = "face"

class SimilarityMethod(str, Enum):
    """相似度计算方法枚举"""
    COSINE = "cosine_similarity"
    EUCLIDEAN = "euclidean_distance"
    DOT_PRODUCT = "dot_product"

# 搜索请求模型
class TextSearchRequest(BaseModel):
    """文本搜索请求"""
    query: str = Field(..., description="搜索查询文本", min_length=1)
    limit: int = Field(10, description="返回结果数量", ge=1, le=100)
    similarity_threshold: float = Field(0.5, description="相似度阈值", ge=0, le=1)
    filters: Optional[Dict[str, Any]] = Field(None, description="搜索过滤条件")
    
    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError('查询文本不能为空')
        return v.strip()

class ImageSearchParams(BaseModel):
    """图像搜索参数"""
    limit: int = Field(10, description="返回结果数量", ge=1, le=100)
    similarity_threshold: float = Field(0.5, description="相似度阈值", ge=0, le=1)
    include_metadata: bool = Field(True, description="是否包含元数据")
    search_faces: bool = Field(False, description="是否同时搜索人脸")

# 人脸识别请求模型
class FaceComparisonRequest(BaseModel):
    """人脸比较请求"""
    face_id_1: str = Field(..., description="第一个人脸ID")
    face_id_2: str = Field(..., description="第二个人脸ID")
    similarity_method: SimilarityMethod = Field(
        SimilarityMethod.COSINE, 
        description="相似度计算方法"
    )

class FaceSearchRequest(BaseModel):
    """人脸搜索请求"""
    limit: int = Field(10, description="返回结果数量", ge=1, le=50)
    similarity_threshold: float = Field(0.8, description="相似度阈值", ge=0, le=1)
    similarity_method: SimilarityMethod = Field(
        SimilarityMethod.COSINE,
        description="相似度计算方法"
    )

# 索引管理请求模型
class IndexBuildRequest(BaseModel):
    """索引构建请求"""
    image_directory: str = Field(..., description="图片目录路径")
    batch_size: int = Field(10, description="批处理大小", ge=1, le=50)
    max_workers: int = Field(4, description="最大并发数", ge=1, le=10)
    force_rebuild: bool = Field(False, description="是否强制重建索引")
    process_faces: bool = Field(True, description="是否处理人脸")

class AddImagesRequest(BaseModel):
    """添加图片请求"""
    image_paths: List[str] = Field(..., description="图片路径列表", min_items=1)
    process_faces: bool = Field(True, description="是否处理人脸")

# 配置请求模型
class ConfigUpdateRequest(BaseModel):
    """配置更新请求"""
    similarity_threshold: Optional[float] = Field(None, ge=0, le=1)
    max_search_results: Optional[int] = Field(None, ge=1, le=100)
    face_similarity_threshold: Optional[float] = Field(None, ge=0, le=1)