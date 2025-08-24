"""
封装 FAISS 数据库的增、查、存、读操作
"""
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class FaissStore:
    """FAISS向量存储类"""

    def __init__(
        self,
        dimension: int = 768,
        index_type: str = "IndexFlatL2",
        metric_type: int = faiss.METRIC_L2,
        **kwargs,
    ):
        """
        初始化FAISS存储

        Args:
            dimension: 向量维度
            index_type: 索引类型
            metric_type: 距离度量类型
            **kwargs: 其他索引参数
        """
        self.dimension = dimension
        self.index_type = index_type
        self.metric_type = metric_type
        self.kwargs = kwargs

        # ID映射：向量索引到原始ID的映射
        self.id_mapping: Dict[int, str] = {}
        self.reverse_id_mapping: Dict[str, int] = {}

        # 创建索引
        self.index = self._create_index()

        logger.info(f"FAISS存储初始化完成: 维度={dimension}, 索引类型={index_type}")

    def _create_index(self) -> faiss.Index:
        """
        创建FAISS索引

        Returns:
            FAISS索引对象
        """
        if self.index_type == "IndexFlatL2":
            return faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "IndexFlatIP":
            return faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "IndexIVFFlat":
            nlist = self.kwargs.get("nlist", 100)
            quantizer = faiss.IndexFlatL2(self.dimension)
            return faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
        else:
            raise ValueError(f"不支持的索引类型: {self.index_type}")

    def add_vectors(
        self, vectors: np.ndarray, ids: List[str], update_existing: bool = True
    ) -> bool:
        """
        添加向量到索引

        Args:
            vectors: 向量数组，形状为 (n, dimension)
            ids: 对应的ID列表
            update_existing: 是否更新已存在的向量

        Returns:
            是否成功添加
        """
        # 验证输入
        if len(vectors) != len(ids):
            raise ValueError(f"向量数量({len(vectors)})与ID数量({len(ids)})不匹配")

        if vectors.shape[1] != self.dimension:
            raise ValueError(f"向量维度({vectors.shape[1]})与索引维度({self.dimension})不匹配")

        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)

        try:
            # 处理重复ID
            new_vectors = []
            new_ids = []
            update_indices = []
            update_vectors = []

            for i, id_str in enumerate(ids):
                if id_str in self.reverse_id_mapping:
                    if update_existing:
                        # 记录需要更新的向量
                        old_index = self.reverse_id_mapping[id_str]
                        update_indices.append(old_index)
                        update_vectors.append(vectors[i])
                    else:
                        logger.warning(f"ID已存在，跳过: {id_str}")
                        continue
                else:
                    new_vectors.append(vectors[i])
                    new_ids.append(id_str)

            # 添加新向量
            if new_vectors:
                new_vectors_array = np.array(new_vectors, dtype=np.float32)

                # 获取当前索引大小
                start_idx = self.index.ntotal

                # 添加到索引
                self.index.add(new_vectors_array)

                # 更新ID映射
                for i, id_str in enumerate(new_ids):
                    idx = start_idx + i
                    self.id_mapping[idx] = id_str
                    self.reverse_id_mapping[id_str] = idx

                logger.info(f"成功添加 {len(new_vectors)} 个新向量")

            # 更新现有向量（FAISS不直接支持更新，需要重建索引）
            if update_vectors and update_existing:
                logger.warning("FAISS不支持原地更新，需要重建索引以更新现有向量")
                # 这里可以实现重建逻辑，或者记录需要更新的向量

            return True

        except Exception as e:
            logger.error(f"添加向量失败: {e}")
            return False

    def search(
        self, query_vectors: np.ndarray, k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        搜索最相似的向量

        Args:
            query_vectors: 查询向量，形状为 (n, dimension)
            k: 返回最相似的k个结果

        Returns:
            (距离数组, 索引数组, ID列表)
        """
        try:
            if len(query_vectors.shape) == 1:
                query_vectors = query_vectors.reshape(1, -1)

            if query_vectors.shape[1] != self.dimension:
                raise ValueError(
                    f"查询向量维度({query_vectors.shape[1]})与索引维度({self.dimension})不匹配"
                )

            if query_vectors.dtype != np.float32:
                query_vectors = query_vectors.astype(np.float32)

            # 限制k值
            k = min(k, self.index.ntotal) if self.index.ntotal > 0 else 0

            if k == 0:
                return np.array([]), np.array([]), []

            # 执行搜索
            distances, indices = self.index.search(query_vectors, k)

            # 转换索引为ID
            result_ids = []
            for i in range(len(query_vectors)):
                query_ids = []
                for j in range(k):
                    idx = indices[i][j]
                    if idx != -1 and idx in self.id_mapping:
                        query_ids.append(self.id_mapping[idx])
                    else:
                        query_ids.append("")
                result_ids.append(query_ids)

            # 如果只有一个查询向量，返回一维结果
            if len(query_vectors) == 1:
                return distances[0], indices[0], result_ids[0]

            return distances, indices, result_ids

        except Exception as e:
            logger.error(f"向量搜索失败: {e}")
            return np.array([]), np.array([]), []

    def save_index(self, index_path: str, metadata_path: Optional[str] = None) -> bool:
        """
        保存索引到文件

        Args:
            index_path: 索引文件路径
            metadata_path: 元数据文件路径，如果为None则自动生成

        Returns:
            是否成功保存
        """
        try:
            # 确保目录存在
            index_path = Path(index_path)
            index_path.parent.mkdir(parents=True, exist_ok=True)

            # 保存FAISS索引
            faiss.write_index(self.index, str(index_path))

            # 保存元数据
            if metadata_path is None:
                metadata_path = index_path.with_suffix(".metadata")

            metadata = {
                "dimension": self.dimension,
                "index_type": self.index_type,
                "metric_type": self.metric_type,
                "kwargs": self.kwargs,
                "id_mapping": self.id_mapping,
                "reverse_id_mapping": self.reverse_id_mapping,
            }

            with open(metadata_path, "wb") as f:
                pickle.dump(metadata, f)

            logger.info(f"索引保存成功: {index_path}")
            return True

        except Exception as e:
            logger.error(f"保存索引失败: {e}")
            return False

    def load_index(self, index_path: str, metadata_path: Optional[str] = None) -> bool:
        """
        从文件加载索引

        Args:
            index_path: 索引文件路径
            metadata_path: 元数据文件路径，如果为None则自动生成

        Returns:
            是否成功加载
        """
        try:
            index_path = Path(index_path)

            if not index_path.exists():
                logger.error(f"加载索引失败: 索引文件不存在: {index_path}")
                raise FileNotFoundError(f"索引文件不存在: {index_path}")

            # 加载元数据
            if metadata_path is None:
                metadata_path = index_path.with_suffix(".metadata")

            if not Path(metadata_path).exists():
                logger.error(f"加载索引失败: 元数据文件不存在: {metadata_path}")
                raise FileNotFoundError(f"元数据文件不存在: {metadata_path}")

            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)

            # 恢复配置
            self.dimension = metadata["dimension"]
            self.index_type = metadata["index_type"]
            self.metric_type = metadata["metric_type"]
            self.kwargs = metadata["kwargs"]
            self.id_mapping = metadata["id_mapping"]
            self.reverse_id_mapping = metadata["reverse_id_mapping"]

            # 加载FAISS索引
            self.index = faiss.read_index(str(index_path))

            logger.info(f"索引加载成功: {index_path}, 向量数量: {self.index.ntotal}")
            return True

        except FileNotFoundError:
            # 重新抛出FileNotFoundError
            raise
        except Exception as e:
            logger.error(f"加载索引失败: {e}")
            return False

    def remove_vector(self, id_str: str) -> bool:
        """
        删除指定ID的向量（注意：FAISS不支持直接删除，需要重建索引）

        Args:
            id_str: 要删除的向量ID

        Returns:
            是否成功删除
        """
        try:
            if id_str not in self.reverse_id_mapping:
                logger.warning(f"向量ID不存在: {id_str}")
                return False

            # FAISS不支持直接删除，这里只是从映射中移除
            # 实际删除需要重建索引
            idx = self.reverse_id_mapping[id_str]
            del self.id_mapping[idx]
            del self.reverse_id_mapping[id_str]

            logger.warning(f"向量ID {id_str} 已从映射中移除，但FAISS索引仍包含该向量")
            logger.warning("要完全删除向量，需要重建索引")

            return True

        except Exception as e:
            logger.error(f"删除向量失败: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取索引统计信息

        Returns:
            统计信息字典
        """
        try:
            stats = {
                "total_vectors": self.index.ntotal,
                "dimension": self.dimension,
                "index_type": self.index_type,
                "metric_type": self.metric_type,
                "id_mapping_size": len(self.id_mapping),
                "reverse_mapping_size": len(self.reverse_id_mapping),
            }

            # 尝试获取内存使用信息（如果可用）
            try:
                if hasattr(self.index, "get_stats"):
                    faiss_stats = self.index.get_stats()
                    stats.update(faiss_stats)
            except:
                pass

            return stats

        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {}

    def rebuild_index(self, remove_ids: Optional[List[str]] = None) -> bool:
        """
        重建索引（用于删除向量或优化索引）

        Args:
            remove_ids: 要删除的ID列表

        Returns:
            是否成功重建
        """
        try:
            if self.index.ntotal == 0:
                logger.info("索引为空，无需重建")
                return True

            # 获取所有向量
            all_vectors = []
            all_ids = []

            for idx in range(self.index.ntotal):
                if idx in self.id_mapping:
                    id_str = self.id_mapping[idx]

                    # 跳过要删除的ID
                    if remove_ids and id_str in remove_ids:
                        continue

                    # 从索引中重构向量（这是一个简化实现）
                    # 实际情况下，你需要保存原始向量或从其他地方获取
                    vector = self.index.reconstruct(idx)
                    all_vectors.append(vector)
                    all_ids.append(id_str)

            if not all_vectors:
                logger.info("重建后索引为空")
                self.index = self._create_index()
                self.id_mapping.clear()
                self.reverse_id_mapping.clear()
                return True

            # 创建新索引
            old_index = self.index
            self.index = self._create_index()
            self.id_mapping.clear()
            self.reverse_id_mapping.clear()

            # 重新添加向量
            vectors_array = np.array(all_vectors, dtype=np.float32)
            success = self.add_vectors(vectors_array, all_ids)

            if not success:
                # 恢复旧索引
                self.index = old_index
                raise Exception("重建索引失败，已恢复旧索引")

            logger.info(f"索引重建成功，向量数量: {len(all_vectors)}")
            return True

        except Exception as e:
            logger.error(f"重建索引失败: {e}")
            return False

    def optimize_index(self) -> bool:
        """
        优化索引（对于某些索引类型）

        Returns:
            是否成功优化
        """
        try:
            # 对于IVF索引，需要训练
            if self.index_type.startswith("IndexIVF"):
                if hasattr(self.index, "is_trained") and not self.index.is_trained:
                    if self.index.ntotal > 0:
                        # 获取训练数据
                        train_vectors = []
                        sample_size = min(1000, self.index.ntotal)

                        for i in range(0, sample_size):
                            vector = self.index.reconstruct(i)
                            train_vectors.append(vector)

                        train_data = np.array(train_vectors, dtype=np.float32)
                        self.index.train(train_data)

                        logger.info("索引训练完成")

            logger.info("索引优化完成")
            return True

        except Exception as e:
            logger.error(f"索引优化失败: {e}")
            return False
