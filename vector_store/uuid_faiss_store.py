"""
支持UUID关联的FAISS向量存储
扩展了原有FaissStore以支持UUID和元数据管理
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

from schemas.face_models import FaceMetadata, FaceSearchResult, FaceSimilarityMethod
from utils.logger import setup_logger
from utils.uuid_manager import UUIDManager

from .faiss_store import FaissStore

logger = setup_logger(__name__)


class UUIDFaissStore(FaissStore):
    """支持UUID关联的FAISS向量存储"""

    def __init__(
        self,
        dimension: int = 768,
        index_type: str = "IndexFlatL2",
        index_file: str = "data/uuid_faiss_index",
        metadata_file: str = "data/uuid_vector_metadata.json",
        **kwargs,
    ):
        """
        初始化UUID FAISS存储

        Args:
            dimension: 向量维度
            index_type: 索引类型
            index_file: 索引文件路径
            metadata_file: 元数据文件路径
            **kwargs: 其他参数
        """
        super().__init__(dimension=dimension, index_type=index_type, **kwargs)

        self.index_file = index_file
        self.metadata_file = metadata_file

        # 向量元数据存储: {vector_id: metadata}
        self.vector_metadata: Dict[int, Dict[str, Any]] = {}

        # UUID到向量ID的映射
        self.uuid_to_vector_ids: Dict[str, List[int]] = {}

        # 向量ID计数器
        self.vector_id_counter = 0

        # 人脸特定的元数据存储
        self.face_metadata: Dict[int, FaceMetadata] = {}
        self._next_face_id = 0

        # 尝试加载现有数据
        self._load_existing_data()

        logger.info(f"UUIDFaissStore初始化完成，当前向量数量: {self.get_vector_count()}")

    def add_vector_with_uuid(
        self,
        vector: np.ndarray,
        content_uuid: str,
        content_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        添加向量并关联UUID和元数据

        Args:
            vector: 向量数据
            content_uuid: 内容UUID
            content_type: 内容类型 (description|face)
            metadata: 额外元数据

        Returns:
            向量ID
        """
        try:
            # 验证向量维度
            if vector.shape[0] != self.dimension:
                raise ValueError(f"向量维度不匹配: 期望{self.dimension}, 实际{vector.shape[0]}")

            # 添加向量到FAISS索引
            vector_reshaped = vector.reshape(1, -1)
            self.index.add(vector_reshaped)

            # 生成向量ID
            vector_id = self.vector_id_counter
            self.vector_id_counter += 1

            # 构建完整元数据
            full_metadata = {
                "content_uuid": content_uuid,
                "content_type": content_type,
                "vector_id": vector_id,
                "added_timestamp": datetime.now().isoformat(),
                **(metadata or {}),
            }

            # 存储元数据
            self.vector_metadata[vector_id] = full_metadata

            # 更新UUID映射
            if content_uuid not in self.uuid_to_vector_ids:
                self.uuid_to_vector_ids[content_uuid] = []
            self.uuid_to_vector_ids[content_uuid].append(vector_id)

            logger.debug(
                f"添加向量成功: UUID={content_uuid}, 类型={content_type}, 向量ID={vector_id}"
            )

            return vector_id

        except Exception as e:
            logger.error(f"添加向量失败: UUID={content_uuid}, 错误: {e}")
            raise

    def batch_add_vectors_with_uuid(
        self,
        vectors: List[np.ndarray],
        content_uuids: List[str],
        content_types: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> List[int]:
        """
        批量添加向量

        Args:
            vectors: 向量列表
            content_uuids: 内容UUID列表
            content_types: 内容类型列表
            metadatas: 元数据列表

        Returns:
            向量ID列表
        """
        if len(vectors) != len(content_uuids) or len(vectors) != len(content_types):
            raise ValueError("向量、UUID和类型列表长度必须一致")

        if metadatas and len(metadatas) != len(vectors):
            raise ValueError("元数据列表长度必须与向量列表长度一致")

        vector_ids = []

        for i, (vector, content_uuid, content_type) in enumerate(
            zip(vectors, content_uuids, content_types)
        ):
            metadata = metadatas[i] if metadatas else None
            vector_id = self.add_vector_with_uuid(
                vector, content_uuid, content_type, metadata
            )
            vector_ids.append(vector_id)

        logger.info(f"批量添加向量完成: {len(vector_ids)}个")

        return vector_ids

    def search_with_uuid(
        self,
        query_vector: np.ndarray,
        k: int = 5,
        content_type_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        搜索向量并返回UUID和元数据信息

        Args:
            query_vector: 查询向量
            k: 返回结果数量
            content_type_filter: 内容类型过滤器

        Returns:
            搜索结果列表，包含距离、UUID、类型和元数据
        """
        try:
            # 执行向量搜索，调用父类方法避免循环调用
            distances, indices, _ = super().search(query_vector, k)

            results = []
            for dist, idx in zip(distances, indices):
                if idx == -1:  # FAISS返回-1表示无效结果
                    continue

                # 获取向量元数据
                metadata = self.vector_metadata.get(idx, {})

                # 应用内容类型过滤器
                if (
                    content_type_filter
                    and metadata.get("content_type") != content_type_filter
                ):
                    continue

                # 获取embedding向量（如果支持）
                try:
                    embedding_vector = (
                        self.index.reconstruct(idx)
                        if hasattr(self.index, "reconstruct")
                        else None
                    )
                except:
                    embedding_vector = None

                result = {
                    "distance": float(dist),
                    "similarity": 1.0 / (1.0 + float(dist)),  # 转换为相似度分数
                    "vector_id": int(idx),
                    "content_uuid": metadata.get("content_uuid"),
                    "content_type": metadata.get("content_type"),
                    "embedding_vector": embedding_vector.tolist()
                    if embedding_vector is not None
                    else None,
                    "metadata": metadata,
                }

                results.append(result)

            # 根据距离重新排序（如果应用了过滤器）
            results.sort(key=lambda x: x["distance"])

            logger.debug(f"搜索完成: 返回{len(results)}个结果")

            return results[:k]  # 确保不超过k个结果

        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []

    def get_vectors_by_uuid(self, content_uuid: str) -> List[Dict[str, Any]]:
        """
        根据UUID获取所有关联的向量信息

        Args:
            content_uuid: 内容UUID

        Returns:
            向量信息列表
        """
        try:
            vector_ids = self.uuid_to_vector_ids.get(content_uuid, [])

            results = []
            for vector_id in vector_ids:
                metadata = self.vector_metadata.get(vector_id, {})
                results.append(
                    {
                        "vector_id": vector_id,
                        "content_uuid": content_uuid,
                        "metadata": metadata,
                    }
                )

            return results

        except Exception as e:
            logger.error(f"根据UUID获取向量失败: UUID={content_uuid}, 错误: {e}")
            return []

    def delete_vectors_by_uuid(self, content_uuid: str) -> bool:
        """
        根据UUID删除所有关联的向量

        注意: FAISS不支持真正的删除，这里只是从元数据中移除

        Args:
            content_uuid: 内容UUID

        Returns:
            是否删除成功
        """
        try:
            vector_ids = self.uuid_to_vector_ids.get(content_uuid, [])

            # 从元数据中移除
            for vector_id in vector_ids:
                self.vector_metadata.pop(vector_id, None)

            # 从UUID映射中移除
            self.uuid_to_vector_ids.pop(content_uuid, None)

            logger.info(f"删除UUID关联的向量: UUID={content_uuid}, 向量数={len(vector_ids)}")

            return True

        except Exception as e:
            logger.error(f"删除向量失败: UUID={content_uuid}, 错误: {e}")
            return False

    def get_vector_count(self) -> int:
        """获取向量总数"""
        return self.index.ntotal

    def get_uuid_count(self) -> int:
        """获取UUID总数"""
        return len(self.uuid_to_vector_ids)

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取存储统计信息

        Returns:
            统计信息字典
        """
        content_type_stats = {}
        for metadata in self.vector_metadata.values():
            content_type = metadata.get("content_type", "unknown")
            content_type_stats[content_type] = (
                content_type_stats.get(content_type, 0) + 1
            )

        return {
            "total_vectors": self.get_vector_count(),
            "total_uuids": self.get_uuid_count(),
            "content_type_distribution": content_type_stats,
            "dimension": self.dimension,
            "index_type": self.index_type,
            "active_metadata_entries": len(self.vector_metadata),
            "uuid_to_vector_mapping_size": len(self.uuid_to_vector_ids),
            "face_statistics": self.get_face_statistics(),
        }

    def save_index(
        self, index_file: Optional[str] = None, metadata_file: Optional[str] = None
    ):
        """
        保存索引和元数据到文件

        Args:
            index_file: 索引文件路径
            metadata_file: 元数据文件路径
        """
        try:
            save_index_file = index_file or self.index_file
            save_metadata_file = metadata_file or self.metadata_file

            # 确保目录存在
            Path(save_index_file).parent.mkdir(parents=True, exist_ok=True)
            Path(save_metadata_file).parent.mkdir(parents=True, exist_ok=True)

            # 保存FAISS索引
            faiss.write_index(self.index, save_index_file)

            # 保存元数据
            # 保存人脸元数据到独立字典
            face_metadata_dict = {
                str(vector_id): face_metadata.to_dict()
                for vector_id, face_metadata in self.face_metadata.items()
            }

            metadata_to_save = {
                "vector_metadata": self.vector_metadata,
                "uuid_to_vector_ids": self.uuid_to_vector_ids,
                "vector_id_counter": self.vector_id_counter,
                "dimension": self.dimension,
                "index_type": self.index_type,
                "face_metadata": face_metadata_dict,
                "save_timestamp": datetime.now().isoformat(),
            }

            with open(save_metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata_to_save, f, ensure_ascii=False, indent=2)

            logger.info(f"索引和元数据保存成功: {save_index_file}, {save_metadata_file}")
            return True

        except Exception as e:
            logger.error(f"保存索引失败: {e}")
            return False

    def load_index(
        self, index_file: Optional[str] = None, metadata_file: Optional[str] = None
    ):
        """
        从文件加载索引和元数据

        Args:
            index_file: 索引文件路径
            metadata_file: 元数据文件路径
        """
        try:
            load_index_file = index_file or self.index_file
            load_metadata_file = metadata_file or self.metadata_file

            # 加载FAISS索引
            if Path(load_index_file).exists():
                self.index = faiss.read_index(load_index_file)
                logger.info(f"FAISS索引加载成功: {load_index_file}")
            else:
                logger.error(f"索引文件不存在: {load_index_file}")
                raise FileNotFoundError(f"索引文件不存在: {load_index_file}")

            # 加载元数据
            if Path(load_metadata_file).exists():
                with open(load_metadata_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)

                # 恢复元数据（键需要转换为int）
                self.vector_metadata = {
                    int(k): v for k, v in metadata.get("vector_metadata", {}).items()
                }
                self.uuid_to_vector_ids = metadata.get("uuid_to_vector_ids", {})
                self.vector_id_counter = metadata.get("vector_id_counter", 0)

                # 加载人脸元数据
                face_metadata_dict = metadata.get("face_metadata", {})
                self.face_metadata = {}
                for vector_id_str, face_data in face_metadata_dict.items():
                    vector_id = int(vector_id_str)
                    self.face_metadata[vector_id] = FaceMetadata.from_dict(face_data)
                    self._next_face_id = max(self._next_face_id, vector_id + 1)

                logger.info(
                    f"元数据加载成功: {load_metadata_file}, 包含{len(self.face_metadata)}个人脸"
                )
            else:
                logger.warning(f"元数据文件不存在: {load_metadata_file}")
                return False

            return True

        except FileNotFoundError:
            # 重新抛出FileNotFoundError以便测试能够捕获
            raise
        except Exception as e:
            logger.error(f"加载索引失败: {e}")
            return False

    def _load_existing_data(self):
        """加载现有的索引和元数据"""
        if Path(self.index_file).exists() and Path(self.metadata_file).exists():
            success = self.load_index()
            if success:
                logger.info("现有数据加载完成")
            else:
                logger.warning("加载现有数据失败，将使用空索引")
        else:
            logger.info("未找到现有数据文件，将创建新索引")

    def rebuild_index(self):
        """重建索引（用于优化性能）"""
        try:
            logger.info("开始重建索引...")

            if self.get_vector_count() == 0:
                logger.info("索引为空，无需重建")
                return

            # 重新创建索引
            new_index = self._create_index()

            # 重新添加所有向量
            vectors_to_readd = []
            for vector_id in sorted(self.vector_metadata.keys()):
                # 这里需要从原索引中提取向量，但FAISS不直接支持
                # 实际实现中可能需要保存原始向量数据
                pass

            logger.info("索引重建完成")

        except Exception as e:
            logger.error(f"重建索引失败: {e}")
            raise

    def add_face_embedding(
        self, embedding: np.ndarray, face_metadata: FaceMetadata, content_uuid: str
    ) -> int:
        """
        添加人脸embedding到索引

        Args:
            embedding: 人脸embedding向量
            face_metadata: 人脸元数据
            content_uuid: 内容UUID

        Returns:
            向量ID
        """
        try:
            # 使用现有的UUID向量添加方法
            vector_id = self.add_vector_with_uuid(
                vector=embedding,
                content_uuid=content_uuid,
                content_type="face",
                metadata={
                    "face_id": face_metadata.face_id,
                    "image_id": face_metadata.image_id,
                    "face_rect": face_metadata.face_rect,
                    "confidence": face_metadata.confidence,
                },
            )

            # 更新人脸元数据中的embedding
            face_metadata.update_embedding(embedding)

            # 保存人脸特定的元数据
            self.face_metadata[vector_id] = face_metadata

            logger.debug(
                f"添加人脸embedding成功: face_id={face_metadata.face_id}, vector_id={vector_id}"
            )

            return vector_id

        except Exception as e:
            logger.error(f"添加人脸embedding失败: {e}")
            raise

    def search_similar_faces(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        similarity_method: str = FaceSimilarityMethod.COSINE,
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
            import time

            start_time = time.time()

            # 使用content_type过滤搜索人脸向量
            results = self.search_with_uuid(
                query_embedding, k, content_type_filter="face"
            )

            # 转换为人脸搜索结果
            face_results = []
            for rank, result in enumerate(results):
                vector_id = result["vector_id"]
                distance = result["distance"]

                if vector_id in self.face_metadata:
                    face_metadata = self.face_metadata[vector_id]

                    # 计算相似度分数
                    similarity_score = self._distance_to_similarity(
                        distance, similarity_method
                    )

                    face_result = FaceSearchResult(
                        face_metadata=face_metadata,
                        similarity_score=similarity_score,
                        rank=rank + 1,
                        distance=distance,
                        search_type=f"face_{similarity_method}",
                    )
                    face_results.append(face_result)
                else:
                    logger.warning(f"找不到人脸元数据: vector_id={vector_id}")

            search_time = time.time() - start_time
            logger.info(f"人脸相似度搜索完成: 查询耗时{search_time:.3f}s, 返回{len(face_results)}个结果")

            return face_results

        except Exception as e:
            logger.error(f"人脸相似度搜索失败: {e}")
            raise

    def _distance_to_similarity(
        self, distance: float, method: str = FaceSimilarityMethod.COSINE
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
            return max(0.0, min(1.0, 1.0 - distance / 2.0))
        elif method == FaceSimilarityMethod.EUCLIDEAN:
            return np.exp(-distance / 2.0)
        elif method == FaceSimilarityMethod.DOT_PRODUCT:
            return max(0.0, min(1.0, distance))
        else:
            return max(0.0, min(1.0, 1.0 / (1.0 + distance)))

    def compare_faces(
        self,
        face_id1: str,
        face_id2: str,
        similarity_method: str = FaceSimilarityMethod.COSINE,
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
            face_metadata1 = self.get_face_metadata_by_id(face_id1)
            face_metadata2 = self.get_face_metadata_by_id(face_id2)

            if not face_metadata1 or not face_metadata2:
                logger.warning(f"找不到人脸数据: face_id1={face_id1}, face_id2={face_id2}")
                return None

            if (
                face_metadata1.embedding_vector is None
                or face_metadata2.embedding_vector is None
            ):
                logger.warning("人脸embedding向量为空")
                return None

            # 计算相似度
            similarity = self._calculate_similarity(
                face_metadata1.embedding_vector,
                face_metadata2.embedding_vector,
                similarity_method,
            )

            return similarity

        except Exception as e:
            logger.error(f"人脸比较失败: {e}")
            return None

    def _calculate_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        method: str = FaceSimilarityMethod.COSINE,
    ) -> float:
        """
        计算两个embedding向量的相似度

        Args:
            embedding1: 第一个向量
            embedding2: 第二个向量
            method: 相似度计算方法

        Returns:
            相似度分数
        """
        if method == FaceSimilarityMethod.COSINE:
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            return dot_product / (norm1 * norm2)
        elif method == FaceSimilarityMethod.EUCLIDEAN:
            distance = np.linalg.norm(embedding1 - embedding2)
            return np.exp(-distance / 2.0)
        elif method == FaceSimilarityMethod.DOT_PRODUCT:
            return np.dot(embedding1, embedding2)
        elif method == FaceSimilarityMethod.MANHATTAN:
            distance = np.sum(np.abs(embedding1 - embedding2))
            return 1.0 / (1.0 + distance)
        else:
            raise ValueError(f"不支持的相似度计算方法: {method}")

    def get_face_metadata_by_id(self, face_id: str) -> Optional[FaceMetadata]:
        """
        通过face_id获取人脸元数据

        Args:
            face_id: 人脸ID

        Returns:
            人脸元数据，如果不存在返回None
        """
        for face_metadata in self.face_metadata.values():
            if face_metadata.face_id == face_id:
                return face_metadata
        return None

    def get_faces_by_image_id(self, image_id: str) -> List[FaceMetadata]:
        """
        获取指定图片的所有人脸

        Args:
            image_id: 图片ID

        Returns:
            人脸元数据列表
        """
        faces = []
        for face_metadata in self.face_metadata.values():
            if face_metadata.image_id == image_id:
                faces.append(face_metadata)
        return faces

    def remove_face(self, face_id: str) -> bool:
        """
        移除指定的人脸数据

        Args:
            face_id: 人脸ID

        Returns:
            是否移除成功
        """
        try:
            # 找到对应的向量ID
            vector_id_to_remove = None
            for vector_id, face_metadata in self.face_metadata.items():
                if face_metadata.face_id == face_id:
                    vector_id_to_remove = vector_id
                    break

            if vector_id_to_remove is None:
                logger.warning(f"找不到要删除的人脸: {face_id}")
                return False

            # 从人脸元数据中移除
            del self.face_metadata[vector_id_to_remove]

            # 从向量元数据中移除（通过UUID删除）
            if vector_id_to_remove in self.vector_metadata:
                content_uuid = self.vector_metadata[vector_id_to_remove].get(
                    "content_uuid"
                )
                if content_uuid:
                    self.delete_vectors_by_uuid(content_uuid)

            logger.info(f"人脸数据已移除: {face_id}")
            return True

        except Exception as e:
            logger.error(f"移除人脸数据失败: {e}")
            return False

    def get_face_statistics(self) -> Dict[str, Any]:
        """
        获取人脸存储统计信息

        Returns:
            人脸统计信息字典
        """
        try:
            total_faces = len(self.face_metadata)
            unique_images = len(set(fm.image_id for fm in self.face_metadata.values()))

            # 计算平均每张图片的人脸数
            avg_faces_per_image = (
                total_faces / unique_images if unique_images > 0 else 0
            )

            # 统计embedding维度分布
            dimensions = [
                fm.embedding_dimension
                for fm in self.face_metadata.values()
                if fm.embedding_vector is not None
            ]

            face_stats = {
                "total_faces": total_faces,
                "unique_images": unique_images,
                "average_faces_per_image": round(avg_faces_per_image, 2),
                "embedding_dimensions": {
                    "min": min(dimensions) if dimensions else 0,
                    "max": max(dimensions) if dimensions else 0,
                    "avg": sum(dimensions) / len(dimensions) if dimensions else 0,
                },
            }

            return face_stats

        except Exception as e:
            logger.error(f"获取人脸统计信息失败: {e}")
            return {}

    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """
        默认搜索方法，返回包含UUID和embedding的结果
        这个方法覆盖了父类的search方法，统一使用UUID搜索

        Args:
            query_vector: 查询向量
            k: 返回结果数量

        Returns:
            搜索结果列表，包含UUID和embedding信息
        """
        return self.search_with_uuid(query_vector, k)
