"""
人脸识别相关数据模型定义
"""
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class FaceMetadata:
    """人脸元数据结构"""

    face_id: str  # 人脸唯一ID
    image_id: str  # 所属图片ID
    image_path: str  # 图片路径
    face_rect: Tuple[int, int, int, int]  # 人脸框坐标 [x,y,w,h]
    confidence_score: float = 0.0  # 检测置信度
    embedding_vector: Optional[np.ndarray] = None  # 人脸embedding向量
    embedding_dimension: int = 1536  # embedding维度
    created_at: Optional[datetime] = None  # 创建时间
    updated_at: Optional[datetime] = None  # 更新时间
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据

    def __post_init__(self):
        """初始化后处理"""
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = self.created_at

    @classmethod
    def create_face_id(cls, image_id: str, face_index: int = 0) -> str:
        """生成人脸ID"""
        return f"{image_id}_face_{face_index}_{uuid.uuid4().hex[:8]}"

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "face_id": self.face_id,
            "image_id": self.image_id,
            "image_path": self.image_path,
            "face_rect": list(self.face_rect),
            "confidence_score": self.confidence_score,
            "embedding_vector": self.embedding_vector.tolist()
            if self.embedding_vector is not None
            else None,
            "embedding_dimension": self.embedding_dimension,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FaceMetadata":
        """从字典创建实例"""
        face_metadata = cls(
            face_id=data["face_id"],
            image_id=data["image_id"],
            image_path=data["image_path"],
            face_rect=tuple(data["face_rect"]),
            confidence_score=data.get("confidence_score", 0.0),
            embedding_dimension=data.get("embedding_dimension", 1536),
            metadata=data.get("metadata", {}),
        )

        # 处理embedding向量
        if data.get("embedding_vector"):
            face_metadata.embedding_vector = np.array(data["embedding_vector"])

        # 处理时间字段
        if data.get("created_at"):
            face_metadata.created_at = datetime.fromisoformat(data["created_at"])
        if data.get("updated_at"):
            face_metadata.updated_at = datetime.fromisoformat(data["updated_at"])

        return face_metadata

    def update_embedding(self, embedding_vector: np.ndarray):
        """更新embedding向量"""
        self.embedding_vector = embedding_vector
        self.embedding_dimension = len(embedding_vector)
        self.updated_at = datetime.now()


@dataclass
class FaceSearchResult:
    """人脸搜索结果数据结构"""

    face_metadata: FaceMetadata
    similarity_score: float  # 相似度分数 [0,1]
    rank: int  # 排名
    distance: float = 0.0  # 向量距离
    search_type: str = "face_similarity"  # 搜索类型

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "face_metadata": self.face_metadata.to_dict(),
            "similarity_score": self.similarity_score,
            "rank": self.rank,
            "distance": self.distance,
            "search_type": self.search_type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FaceSearchResult":
        """从字典创建实例"""
        return cls(
            face_metadata=FaceMetadata.from_dict(data["face_metadata"]),
            similarity_score=data["similarity_score"],
            rank=data["rank"],
            distance=data.get("distance", 0.0),
            search_type=data.get("search_type", "face_similarity"),
        )


@dataclass
class FaceComparisonResult:
    """人脸比较结果"""

    face1_id: str
    face2_id: str
    similarity_score: float  # 相似度分数 [0,1]
    distance: float  # 向量距离
    is_same_person: bool  # 是否是同一人 (基于阈值)
    confidence_threshold: float = 0.8  # 判断阈值
    comparison_method: str = "cosine_similarity"  # 比较方法

    def __post_init__(self):
        """初始化后处理"""
        self.is_same_person = self.similarity_score >= self.confidence_threshold

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "face1_id": self.face1_id,
            "face2_id": self.face2_id,
            "similarity_score": self.similarity_score,
            "distance": self.distance,
            "is_same_person": self.is_same_person,
            "confidence_threshold": self.confidence_threshold,
            "comparison_method": self.comparison_method,
        }


@dataclass
class FaceBatchProcessingResult:
    """批量人脸处理结果"""

    total_images: int
    total_faces: int
    successful_extractions: int
    failed_extractions: int
    processing_time_seconds: float
    error_messages: List[str] = field(default_factory=list)
    face_metadata_list: List[FaceMetadata] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.total_images == 0:
            return 0.0
        return self.successful_extractions / self.total_images

    def add_error(self, error_message: str):
        """添加错误信息"""
        self.error_messages.append(error_message)
        self.failed_extractions += 1

    def add_face_metadata(self, face_metadata: FaceMetadata):
        """添加人脸元数据"""
        self.face_metadata_list.append(face_metadata)
        self.total_faces += 1

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "total_images": self.total_images,
            "total_faces": self.total_faces,
            "successful_extractions": self.successful_extractions,
            "failed_extractions": self.failed_extractions,
            "success_rate": self.success_rate,
            "processing_time_seconds": self.processing_time_seconds,
            "error_messages": self.error_messages,
            "face_count": len(self.face_metadata_list),
        }


class FaceSimilarityMethod:
    """人脸相似度计算方法枚举"""

    COSINE = "cosine_similarity"
    EUCLIDEAN = "euclidean_distance"
    DOT_PRODUCT = "dot_product"
    MANHATTAN = "manhattan_distance"


class FaceProcessingStatus:
    """人脸处理状态枚举"""

    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class FaceProcessorConfig:
    """人脸处理器配置"""

    embedding_dimension: int = 1536
    max_batch_size: int = 10
    enable_parallel_processing: bool = True
    face_detection_threshold: float = 0.5
    similarity_threshold: float = 0.8
    max_faces_per_image: int = 10
    processing_timeout_seconds: int = 30

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "embedding_dimension": self.embedding_dimension,
            "max_batch_size": self.max_batch_size,
            "enable_parallel_processing": self.enable_parallel_processing,
            "face_detection_threshold": self.face_detection_threshold,
            "similarity_threshold": self.similarity_threshold,
            "max_faces_per_image": self.max_faces_per_image,
            "processing_timeout_seconds": self.processing_timeout_seconds,
        }


@dataclass
class FaceIndexConfig:
    """人脸索引配置"""

    index_type: str = "Flat"
    metric: str = "IP"  # Inner Product
    nlist: int = 100
    nprobe: int = 10
    dimension: int = 1536
    enable_gpu: bool = False
    index_file_path: str = "data/face_faiss_index"
    metadata_file_path: str = "data/face_metadata.json"

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "index_type": self.index_type,
            "metric": self.metric,
            "nlist": self.nlist,
            "nprobe": self.nprobe,
            "dimension": self.dimension,
            "enable_gpu": self.enable_gpu,
            "index_file_path": self.index_file_path,
            "metadata_file_path": self.metadata_file_path,
        }


# 默认配置实例
default_face_processor_config = FaceProcessorConfig()
default_face_index_config = FaceIndexConfig()
