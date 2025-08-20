"""
核心数据模型定义
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime
from enum import Enum


class ProcessingStatus(Enum):
    """处理状态枚举"""
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"
    PENDING = "pending"


@dataclass
class ImageMetadata:
    """图片元数据结构"""
    path: str  # 原图绝对路径
    is_snap: bool = False  # 是否是手机截图
    is_landscape: bool = False  # 是否是风景照
    description: str = ""  # 对当前的描述，用于语义检索
    has_person: bool = False  # 是否有人
    face_rects: List[Tuple[int, int, int, int]] = field(default_factory=list)  # 人脸框 [x,y,w,h]
    timestamp: Optional[str] = None  # 照片的时间戳
    unique_id: str = ""  # 获取唯一 ID
    processing_status: ProcessingStatus = ProcessingStatus.PENDING  # 处理状态
    error_message: str = ""  # 错误信息
    retry_count: int = 0  # 重试次数
    last_processed: Optional[datetime] = None  # 最后处理时间
    face_processing_info: Optional[Dict[str, Any]] = None  # 人脸处理信息
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "path": self.path,
            "is_snap": self.is_snap,
            "is_landscape": self.is_landscape,
            "description": self.description,
            "has_person": self.has_person,
            "face_rects": self.face_rects,
            "timestamp": self.timestamp,
            "unique_id": self.unique_id,
            "processing_status": self.processing_status.value,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
            "last_processed": self.last_processed.isoformat() if self.last_processed else None,
            "face_processing_info": self.face_processing_info
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ImageMetadata":
        """从字典创建实例"""
        instance = cls(
            path=data["path"],
            is_snap=data.get("is_snap", False),
            is_landscape=data.get("is_landscape", False),
            description=data.get("description", ""),
            has_person=data.get("has_person", False),
            face_rects=data.get("face_rects", []),
            timestamp=data.get("timestamp"),
            unique_id=data.get("unique_id", ""),
            processing_status=ProcessingStatus(data.get("processing_status", "pending")),
            error_message=data.get("error_message", ""),
            retry_count=data.get("retry_count", 0),
            face_processing_info=data.get("face_processing_info")
        )
        
        if data.get("last_processed"):
            instance.last_processed = datetime.fromisoformat(data["last_processed"])
        
        return instance


@dataclass
class SearchResult:
    """搜索结果数据结构"""
    metadata: ImageMetadata
    similarity_score: float
    rank: int
    search_type: str  # "semantic" | "face" | "rerank"