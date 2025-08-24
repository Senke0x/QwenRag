"""
API响应数据模型
"""
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


# 基础响应模型
class BaseResponse(BaseModel):
    """基础响应模型"""

    success: bool = Field(True, description="请求是否成功")
    message: str = Field("操作成功", description="响应消息")
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())


class ErrorResponse(BaseResponse):
    """错误响应模型"""

    success: bool = Field(False, description="请求失败")
    error_code: str = Field(..., description="错误代码")
    error_detail: Optional[str] = Field(None, description="错误详情")


# 搜索结果模型
class ImageMetadataResponse(BaseModel):
    """图片元数据响应"""

    image_id: str = Field(..., description="图片ID")
    image_path: str = Field(..., description="图片路径")
    description: str = Field("", description="图片描述")
    is_snap: bool = Field(False, description="是否截图")
    is_landscape: bool = Field(False, description="是否风景照")
    has_person: bool = Field(False, description="是否有人物")
    face_count: int = Field(0, description="人脸数量")
    timestamp: Optional[str] = Field(None, description="时间戳")
    additional_metadata: Optional[Dict[str, Any]] = Field(None, description="额外元数据")


class SearchResultItem(BaseModel):
    """搜索结果项"""

    image_metadata: ImageMetadataResponse
    similarity_score: float = Field(..., description="相似度分数", ge=0, le=1)
    rank: int = Field(..., description="排名", ge=1)
    search_type: str = Field(..., description="搜索类型")


class SearchResponse(BaseResponse):
    """搜索响应基类"""

    results: List[SearchResultItem] = Field([], description="搜索结果")
    total_results: int = Field(0, description="总结果数")
    query_time_ms: float = Field(..., description="查询耗时(毫秒)")
    search_params: Dict[str, Any] = Field({}, description="搜索参数")


class TextSearchResponse(SearchResponse):
    """文本搜索响应"""

    query: str = Field(..., description="搜索查询")


class ImageSearchResponse(SearchResponse):
    """图像搜索响应"""

    uploaded_image_info: Optional[Dict[str, Any]] = Field(None, description="上传图片信息")


# 人脸识别响应模型
class DetectedFace(BaseModel):
    """检测到的人脸"""

    face_id: str = Field(..., description="人脸ID")
    bounding_box: List[int] = Field(..., description="人脸边界框 [x, y, w, h]")
    confidence: float = Field(..., description="检测置信度", ge=0, le=1)
    embedding_vector: Optional[List[float]] = Field(None, description="人脸特征向量")


class FaceDetectionResponse(BaseResponse):
    """人脸检测响应"""

    faces: List[DetectedFace] = Field([], description="检测到的人脸列表")
    face_count: int = Field(0, description="人脸数量")
    processing_time_ms: float = Field(..., description="处理耗时(毫秒)")
    image_info: Optional[Dict[str, Any]] = Field(None, description="图片信息")


class FaceSearchResultItem(BaseModel):
    """人脸搜索结果项"""

    face_id: str = Field(..., description="人脸ID")
    image_id: str = Field(..., description="所属图片ID")
    image_path: str = Field(..., description="图片路径")
    face_bounding_box: List[int] = Field(..., description="人脸边界框")
    similarity_score: float = Field(..., description="相似度分数", ge=0, le=1)
    rank: int = Field(..., description="排名", ge=1)


class FaceSearchResponse(BaseResponse):
    """人脸搜索响应"""

    results: List[FaceSearchResultItem] = Field([], description="搜索结果")
    total_results: int = Field(0, description="总结果数")
    query_time_ms: float = Field(..., description="查询耗时(毫秒)")
    search_params: Dict[str, Any] = Field({}, description="搜索参数")


class FaceComparisonResponse(BaseResponse):
    """人脸比较响应"""

    face1_id: str = Field(..., description="第一个人脸ID")
    face2_id: str = Field(..., description="第二个人脸ID")
    similarity_score: float = Field(..., description="相似度分数", ge=0, le=1)
    distance: float = Field(..., description="向量距离")
    is_same_person: bool = Field(..., description="是否同一人")
    confidence_threshold: float = Field(..., description="判断阈值")
    comparison_method: str = Field(..., description="比较方法")


# 索引管理响应模型
class IndexStatusResponse(BaseResponse):
    """索引状态响应"""

    total_images: int = Field(0, description="总图片数")
    indexed_images: int = Field(0, description="已索引图片数")
    total_faces: int = Field(0, description="总人脸数")
    index_size_mb: float = Field(0, description="索引大小(MB)")
    last_updated: Optional[str] = Field(None, description="最后更新时间")
    index_health: str = Field("unknown", description="索引健康状态")


class IndexBuildResponse(BaseResponse):
    """索引构建响应"""

    task_id: str = Field(..., description="任务ID")
    status: str = Field(..., description="任务状态")
    total_images: int = Field(0, description="总图片数")
    processed_images: int = Field(0, description="已处理图片数")
    failed_images: int = Field(0, description="失败图片数")
    processing_time_seconds: float = Field(0, description="处理耗时(秒)")
    face_processing_results: Optional[Dict[str, Any]] = Field(
        None, description="人脸处理结果"
    )


class AddImagesResponse(BaseResponse):
    """添加图片响应"""

    added_images: int = Field(0, description="添加的图片数")
    failed_images: int = Field(0, description="失败的图片数")
    processing_time_seconds: float = Field(0, description="处理耗时(秒)")
    image_details: List[Dict[str, Any]] = Field([], description="图片处理详情")


# 统计信息响应模型
class SystemStatsResponse(BaseResponse):
    """系统统计响应"""

    total_images: int = Field(0, description="总图片数")
    total_faces: int = Field(0, description="总人脸数")
    index_statistics: Dict[str, Any] = Field({}, description="索引统计")
    face_statistics: Dict[str, Any] = Field({}, description="人脸统计")
    system_info: Dict[str, Any] = Field({}, description="系统信息")
    uptime_seconds: float = Field(0, description="运行时间(秒)")


# 文件上传响应模型
class FileUploadResponse(BaseResponse):
    """文件上传响应"""

    file_id: str = Field(..., description="文件ID")
    file_path: str = Field(..., description="文件路径")
    file_size: int = Field(..., description="文件大小(字节)")
    file_type: str = Field(..., description="文件类型")
    upload_time: str = Field(..., description="上传时间")
    expires_at: Optional[str] = Field(None, description="过期时间")
