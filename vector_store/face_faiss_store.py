"""
专门的人脸向量存储模块
基于FAISS实现高效的人脸相似度搜索
"""
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import faiss
import numpy as np

from schemas.face_models import FaceMetadata, FaceSearchResult, FaceSimilarityMethod
from vector_store.faiss_store import FaissStore
from utils.logger import setup_logger

logger = setup_logger(__name__)


class FaceFaissStore(FaissStore):
    """专门的人脸向量存储，继承基础FaissStore"""
    
    def __init__(
        self,
        dimension: int = 1536,
        index_type: str = "IndexFlatIP",
        index_path: Optional[str] = None,
        metadata_path: Optional[str] = None
    ):
        """
        初始化人脸向量存储
        
        Args:
            dimension: 向量维度
            index_type: 索引类型 ("IndexFlatL2", "IndexFlatIP", "IndexIVFFlat")
            index_path: 索引文件路径
            metadata_path: 元数据文件路径
        """
        # 使用专门的人脸索引路径
        if index_path is None:
            index_path = "data/face_faiss_index"
        if metadata_path is None:
            metadata_path = "data/face_metadata.json"
            
        super().__init__(dimension, index_type)
        
        self.metadata_path = Path(metadata_path)
        self.face_metadata: Dict[int, FaceMetadata] = {}
        self._next_face_id = 0
        
        # 加载现有的人脸元数据
        self._load_face_metadata()
        
        logger.info(f"人脸向量存储初始化完成: dimension={dimension}, "
                   f"index_type={index_type}, 现有人脸数={len(self.face_metadata)}")
    
    def _load_face_metadata(self):
        """加载人脸元数据"""
        try:
            if self.metadata_path.exists():
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    metadata_data = json.load(f)
                    
                # 转换为FaceMetadata对象
                for face_idx_str, face_data in metadata_data.items():
                    face_idx = int(face_idx_str)
                    self.face_metadata[face_idx] = FaceMetadata.from_dict(face_data)
                    self._next_face_id = max(self._next_face_id, face_idx + 1)
                
                logger.info(f"已加载 {len(self.face_metadata)} 个人脸元数据")
        except Exception as e:
            logger.warning(f"加载人脸元数据失败: {e}")
            self.face_metadata = {}
            self._next_face_id = 0
    
    def _save_face_metadata(self):
        """保存人脸元数据到文件"""
        try:
            # 确保目录存在
            self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 转换为可序列化的字典
            metadata_dict = {
                str(face_idx): face_metadata.to_dict()
                for face_idx, face_metadata in self.face_metadata.items()
            }
            
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata_dict, f, ensure_ascii=False, indent=2)
                
            logger.debug(f"人脸元数据已保存到: {self.metadata_path}")
        except Exception as e:
            logger.error(f"保存人脸元数据失败: {e}")
            raise
    
    def add_face_embedding(
        self, 
        embedding: np.ndarray, 
        face_metadata: FaceMetadata
    ) -> int:
        """
        添加人脸embedding到索引
        
        Args:
            embedding: 人脸embedding向量
            face_metadata: 人脸元数据
            
        Returns:
            分配的索引ID
        """
        try:
            # 验证embedding维度
            if len(embedding) != self.dimension:
                raise ValueError(f"Embedding维度不匹配: 期望{self.dimension}, 实际{len(embedding)}")
            
            # 添加向量到索引
            face_idx = self.add_vector(embedding)
            
            # 更新人脸元数据中的embedding
            face_metadata.update_embedding(embedding)
            
            # 保存元数据
            self.face_metadata[face_idx] = face_metadata
            
            # 立即保存元数据文件
            self._save_face_metadata()
            
            logger.debug(f"添加人脸embedding成功: face_id={face_metadata.face_id}, "
                        f"index_id={face_idx}")
            
            return face_idx
            
        except Exception as e:
            logger.error(f"添加人脸embedding失败: {e}")
            raise
    
    def search_similar_faces(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        similarity_method: str = FaceSimilarityMethod.COSINE
    ) -> List[FaceSearchResult]:
        """
        搜索相似人脸
        
        Args:
            query_embedding: 查询人脸的embedding
            k: 返回结果数量
            similarity_method: 相似度计算方法
            
        Returns:
            人脸搜索结果列表
        """
        try:
            start_time = time.time()
            
            # 使用父类的搜索方法
            search_results = self.search(query_embedding, k)
            
            # 转换为人脸搜索结果
            face_results = []
            for rank, (face_idx, distance) in enumerate(search_results):
                if face_idx in self.face_metadata:
                    face_metadata = self.face_metadata[face_idx]
                    
                    # 计算相似度分数
                    similarity_score = self._distance_to_similarity(
                        distance, similarity_method
                    )
                    
                    face_result = FaceSearchResult(
                        face_metadata=face_metadata,
                        similarity_score=similarity_score,
                        rank=rank + 1,
                        distance=distance,
                        search_type=f"face_{similarity_method}"
                    )
                    face_results.append(face_result)
                else:
                    logger.warning(f"找不到人脸元数据: face_idx={face_idx}")
            
            search_time = time.time() - start_time
            logger.info(f"人脸相似度搜索完成: 查询耗时{search_time:.3f}s, "
                       f"返回{len(face_results)}个结果")
            
            return face_results
            
        except Exception as e:
            logger.error(f"人脸相似度搜索失败: {e}")
            raise
    
    def _distance_to_similarity(
        self, 
        distance: float, 
        method: str = FaceSimilarityMethod.COSINE
    ) -> float:
        """
        将距离转换为相似度分数 [0,1]
        
        Args:
            distance: 向量距离
            method: 相似度计算方法
            
        Returns:
            相似度分数
        """
        if method == FaceSimilarityMethod.COSINE:
            # 对于余弦距离，相似度 = 1 - distance/2
            # 确保结果在[0,1]范围内
            return max(0.0, min(1.0, 1.0 - distance / 2.0))
        
        elif method == FaceSimilarityMethod.EUCLIDEAN:
            # 对于欧几里得距离，使用指数衰减
            return np.exp(-distance / 2.0)
        
        elif method == FaceSimilarityMethod.DOT_PRODUCT:
            # 点积已经是相似度
            return max(0.0, min(1.0, distance))
        
        else:
            # 默认处理
            return max(0.0, min(1.0, 1.0 / (1.0 + distance)))
    
    def compare_faces(
        self,
        face_id1: str,
        face_id2: str,
        similarity_method: str = FaceSimilarityMethod.COSINE
    ) -> Optional[float]:
        """
        比较两个人脸的相似度
        
        Args:
            face_id1: 第一个人脸ID
            face_id2: 第二个人脸ID  
            similarity_method: 相似度计算方法
            
        Returns:
            相似度分数，如果找不到人脸则返回None
        """
        try:
            # 查找两个人脸的embedding
            face_metadata1 = self.get_face_metadata_by_id(face_id1)
            face_metadata2 = self.get_face_metadata_by_id(face_id2)
            
            if not face_metadata1 or not face_metadata2:
                logger.warning(f"找不到人脸数据: face_id1={face_id1}, face_id2={face_id2}")
                return None
            
            if (face_metadata1.embedding_vector is None or 
                face_metadata2.embedding_vector is None):
                logger.warning("人脸embedding向量为空")
                return None
            
            # 计算相似度
            similarity = self._calculate_similarity(
                face_metadata1.embedding_vector,
                face_metadata2.embedding_vector,
                similarity_method
            )
            
            return similarity
            
        except Exception as e:
            logger.error(f"人脸比较失败: {e}")
            return None
    
    def _calculate_similarity(
        self, 
        embedding1: np.ndarray, 
        embedding2: np.ndarray,
        method: str = FaceSimilarityMethod.COSINE
    ) -> float:
        """计算两个embedding向量的相似度"""
        if method == FaceSimilarityMethod.COSINE:
            # 余弦相似度
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            return dot_product / (norm1 * norm2)
            
        elif method == FaceSimilarityMethod.EUCLIDEAN:
            # 欧几里得距离转相似度
            distance = np.linalg.norm(embedding1 - embedding2)
            return np.exp(-distance / 2.0)
            
        elif method == FaceSimilarityMethod.DOT_PRODUCT:
            # 点积
            return np.dot(embedding1, embedding2)
            
        elif method == FaceSimilarityMethod.MANHATTAN:
            # 曼哈顿距离转相似度
            distance = np.sum(np.abs(embedding1 - embedding2))
            return 1.0 / (1.0 + distance)
            
        else:
            raise ValueError(f"不支持的相似度计算方法: {method}")
    
    def get_face_metadata_by_id(self, face_id: str) -> Optional[FaceMetadata]:
        """通过face_id获取人脸元数据"""
        for face_metadata in self.face_metadata.values():
            if face_metadata.face_id == face_id:
                return face_metadata
        return None
    
    def get_faces_by_image_id(self, image_id: str) -> List[FaceMetadata]:
        """获取指定图片的所有人脸"""
        faces = []
        for face_metadata in self.face_metadata.values():
            if face_metadata.image_id == image_id:
                faces.append(face_metadata)
        return faces
    
    def remove_face(self, face_id: str) -> bool:
        """移除指定的人脸数据"""
        try:
            # 找到对应的索引
            face_idx_to_remove = None
            for face_idx, face_metadata in self.face_metadata.items():
                if face_metadata.face_id == face_id:
                    face_idx_to_remove = face_idx
                    break
            
            if face_idx_to_remove is None:
                logger.warning(f"找不到要删除的人脸: {face_id}")
                return False
            
            # 从元数据中移除
            del self.face_metadata[face_idx_to_remove]
            
            # TODO: 从FAISS索引中移除向量（FAISS不直接支持删除，需要重建索引）
            logger.warning("FAISS不支持直接删除向量，建议定期重建索引")
            
            # 保存更新后的元数据
            self._save_face_metadata()
            
            logger.info(f"人脸数据已移除: {face_id}")
            return True
            
        except Exception as e:
            logger.error(f"移除人脸数据失败: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取人脸存储统计信息"""
        try:
            total_faces = len(self.face_metadata)
            unique_images = len(set(fm.image_id for fm in self.face_metadata.values()))
            
            # 计算平均每张图片的人脸数
            avg_faces_per_image = total_faces / unique_images if unique_images > 0 else 0
            
            # 统计embedding维度分布
            dimensions = [fm.embedding_dimension for fm in self.face_metadata.values() 
                         if fm.embedding_vector is not None]
            
            stats = {
                "total_faces": total_faces,
                "unique_images": unique_images,
                "average_faces_per_image": round(avg_faces_per_image, 2),
                "index_dimension": self.dimension,
                "index_type": self.index_type,
                "metadata_file_size": self.metadata_path.stat().st_size if self.metadata_path.exists() else 0,
                "embedding_dimensions": {
                    "min": min(dimensions) if dimensions else 0,
                    "max": max(dimensions) if dimensions else 0,
                    "avg": sum(dimensions) / len(dimensions) if dimensions else 0
                }
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {}
    
    def rebuild_index(self):
        """重建整个人脸索引"""
        try:
            logger.info("开始重建人脸索引...")
            
            # 收集所有有效的embedding
            valid_embeddings = []
            valid_metadata = {}
            
            for old_idx, face_metadata in self.face_metadata.items():
                if face_metadata.embedding_vector is not None:
                    valid_embeddings.append(face_metadata.embedding_vector)
                    valid_metadata[len(valid_embeddings) - 1] = face_metadata
            
            if not valid_embeddings:
                logger.warning("没有有效的人脸embedding，跳过重建")
                return
            
            # 重新初始化索引
            self._init_index()
            
            # 批量添加向量
            embeddings_array = np.array(valid_embeddings).astype('float32')
            self.index.add(embeddings_array)
            
            # 更新元数据映射
            self.face_metadata = valid_metadata
            self._next_face_id = len(valid_metadata)
            
            # 保存更新后的索引和元数据
            self.save_index()
            self._save_face_metadata()
            
            logger.info(f"人脸索引重建完成: {len(valid_embeddings)} 个向量")
            
        except Exception as e:
            logger.error(f"重建人脸索引失败: {e}")
            raise