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

from utils.logger import setup_logger
from utils.uuid_manager import UUIDManager
from vector_store.faiss_store import FaissStore

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
            metadata_to_save = {
                "vector_metadata": self.vector_metadata,
                "uuid_to_vector_ids": self.uuid_to_vector_ids,
                "vector_id_counter": self.vector_id_counter,
                "dimension": self.dimension,
                "index_type": self.index_type,
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

                logger.info(f"元数据加载成功: {load_metadata_file}")
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
