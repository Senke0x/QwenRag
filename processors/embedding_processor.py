"""
Embedding处理器 - 负责向量化和存储管理
"""
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from clients.qwen_client import QwenClient
from config.embedding_config import EmbeddingProcessorConfig, default_embedding_config
from utils.logger import logger
from vector_store.faiss_store import FaissStore

logger = logging.getLogger(__name__)


class EmbeddingProcessor:
    """
    Embedding处理器 - 负责向量化和存储管理
    支持两种embedding模式：文本和图片base64
    """

    def __init__(
        self,
        qwen_client: Optional[QwenClient] = None,
        vector_store: Optional[FaissStore] = None,
        config: Optional[EmbeddingProcessorConfig] = None,
    ):
        """
        初始化Embedding处理器

        Args:
            qwen_client: Qwen客户端实例
            vector_store: FAISS存储实例
            config: EmbeddingProcessor配置
        """
        self.config = config or default_embedding_config
        self.config.validate()

        self.qwen_client = qwen_client or QwenClient()

        # 初始化embedding维度
        if self.config.embedding_dimension is None:
            self._detected_dimension = self._detect_embedding_dimension()
            self.embedding_dimension = self._detected_dimension
        else:
            self.embedding_dimension = self.config.embedding_dimension

        self.vector_store = vector_store or FaissStore(
            dimension=self.embedding_dimension
        )
        self.index_save_path = self.config.index_save_path

        # 确保保存目录存在
        Path(self.index_save_path).parent.mkdir(parents=True, exist_ok=True)

        # 加载已有索引（如果存在）
        self._load_existing_index()

        logger.info(f"EmbeddingProcessor初始化完成，维度: {self.embedding_dimension}")

    def _detect_embedding_dimension(self) -> int:
        """
        自动检测embedding维度

        Returns:
            检测到的embedding维度
        """
        try:
            logger.info("自动检测embedding维度...")
            response = self.qwen_client.get_text_embedding("dimension detection test")
            dimension = len(response["embedding"])
            logger.info(f"检测到embedding维度: {dimension}")
            return dimension
        except Exception as e:
            logger.warning(f"维度检测失败，使用默认维度: {e}")
            return self.config.default_embedding_dimension

    def _validate_dimension(self, vector: np.ndarray, context: str) -> bool:
        """
        验证向量维度一致性

        Args:
            vector: 待验证的向量
            context: 上下文信息

        Returns:
            是否维度一致
        """
        if vector.shape[0] != self.embedding_dimension:
            logger.error(
                f"{context} 向量维度不匹配: 期望{self.embedding_dimension}, "
                f"实际{vector.shape[0]}"
            )
            return False
        return True

    def _load_existing_index(self) -> bool:
        """
        加载已存在的向量索引

        Returns:
            是否成功加载
        """
        try:
            # 尝试加载已有索引
            success = self.vector_store.load_index(self.index_save_path)
            if success:
                logger.info(f"成功加载已有索引，向量数量: {self.vector_store.index.ntotal}")
                return True
            return False
        except FileNotFoundError:
            logger.info("未找到已有索引，将创建新索引")
            return False
        except Exception as e:
            logger.error(f"加载索引失败: {e}")
            return False

    def process_text(self, text: str, text_id: str) -> bool:
        """
        处理文本数据，生成并存储向量

        Args:
            text: 文本内容
            text_id: 文本唯一标识

        Returns:
            是否处理成功
        """
        try:
            logger.info(f"处理文本数据: {text_id}")

            # 使用text-embedding-v4处理文本
            response = self.qwen_client.get_text_embedding(text)
            text_vector = np.array(response["embedding"], dtype=np.float32)

            # 验证维度一致性
            if not self._validate_dimension(text_vector, "文本处理"):
                return False

            # 存储向量
            success = self.vector_store.add_vectors(
                vectors=text_vector.reshape(1, -1), ids=[text_id]
            )

            if success:
                logger.info(f"成功处理文本数据: {text_id}")
                if self.config.auto_save:
                    self.save_index()
                return True
            else:
                logger.error(f"文本数据处理失败: {text_id}")
                return False

        except Exception as e:
            logger.error(f"处理文本数据失败: {text_id}, 错误: {e}")
            return False

    def process_image_base64(self, image_base64: str, image_id: str) -> bool:
        """
        处理base64图片数据，生成并存储向量

        Args:
            image_base64: 图片base64编码
            image_id: 图片唯一标识

        Returns:
            是否处理成功
        """
        try:
            logger.info(f"处理base64图片: {image_id}")

            # 使用multimodal-embedding-v1处理图片
            response = self.qwen_client.get_image_embedding(image_base64)
            image_vector = np.array(response["embedding"], dtype=np.float32)

            # 验证维度一致性
            if not self._validate_dimension(image_vector, "图片处理"):
                return False

            # 存储向量
            success = self.vector_store.add_vectors(
                vectors=image_vector.reshape(1, -1), ids=[image_id]
            )

            if success:
                logger.info(f"成功处理base64图片: {image_id}")
                if self.config.auto_save:
                    self.save_index()
                return True
            else:
                logger.error(f"base64图片处理失败: {image_id}")
                return False

        except Exception as e:
            logger.error(f"处理base64图片失败: {image_id}, 错误: {e}")
            return False

    def process_image_with_faces(
        self, image_base64: str, face_rects: List[Dict[str, int]], image_id: str
    ) -> bool:
        """
        处理包含人脸的图片，为每个人脸区域生成embedding

        Args:
            image_base64: 原图片base64编码
            face_rects: 人脸矩形区域列表
            image_id: 图片唯一标识

        Returns:
            是否处理成功
        """
        try:
            logger.info(f"处理包含{len(face_rects)}个人脸的图片: {image_id}")

            vectors_to_store = []
            ids_to_store = []

            # 处理整张图片
            whole_image_result = self.qwen_client.get_image_embedding(image_base64)
            whole_image_vector = np.array(
                whole_image_result["embedding"], dtype=np.float32
            )
            vectors_to_store.append(whole_image_vector)
            ids_to_store.append(f"{image_id}_full")

            # 处理每个人脸区域
            for i, face_rect in enumerate(face_rects):
                try:
                    face_result = self.qwen_client.get_face_embedding(
                        image_base64, face_rect
                    )
                    face_vector = np.array(face_result["embedding"], dtype=np.float32)
                    vectors_to_store.append(face_vector)
                    ids_to_store.append(f"{image_id}_face_{i}")
                    logger.debug(f"成功处理人脸{i}: {image_id}")
                except Exception as e:
                    logger.warning(f"处理人脸{i}失败: {e}")
                    # 继续处理其他人脸
                    continue

            # 验证维度一致性
            if vectors_to_store and not self._validate_dimension(
                vectors_to_store[0], "人脸处理"
            ):
                return False

            # 批量存储向量
            if vectors_to_store:
                vectors_array = np.vstack(vectors_to_store)
                success = self.vector_store.add_vectors(vectors_array, ids_to_store)

                if success:
                    logger.info(f"成功存储 {len(vectors_to_store)} 个向量: {image_id}")
                    if self.config.auto_save:
                        self.save_index()
                    return True
                else:
                    logger.error(f"向量存储失败: {image_id}")
                    return False
            else:
                logger.warning(f"没有生成向量数据: {image_id}")
                return False

        except Exception as e:
            logger.error(f"处理带人脸图片失败: {image_id}, 错误: {e}")
            return False

    def _process_single_text(self, text: str, text_id: str) -> bool:
        """
        处理单个文本（不自动保存索引，用于批量处理）

        Args:
            text: 文本内容
            text_id: 文本唯一标识

        Returns:
            是否处理成功
        """
        try:
            response = self.qwen_client.get_text_embedding(text)
            text_vector = np.array(response["embedding"], dtype=np.float32)

            # 验证维度一致性
            if not self._validate_dimension(text_vector, f"文本处理-{text_id}"):
                return False

            # 存储向量（不自动保存索引）
            success = self.vector_store.add_vectors(
                vectors=text_vector.reshape(1, -1), ids=[text_id]
            )

            if success:
                logger.debug(f"成功处理文本数据: {text_id}")
                return True
            else:
                logger.error(f"文本数据处理失败: {text_id}")
                return False

        except Exception as e:
            logger.error(f"处理文本数据失败: {text_id}, 错误: {e}")
            return False

    def _process_text_batch(
        self, texts: List[str], text_ids: List[str]
    ) -> Dict[str, Any]:
        """
        内部批量处理文本的实际实现

        Args:
            texts: 文本列表
            text_ids: 文本ID列表

        Returns:
            处理结果统计
        """
        results = {"success": 0, "failed": 0, "failed_items": []}

        # 使用线程池并行处理
        if self.config.enable_parallel_processing and len(texts) > 1:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                # 提交所有任务
                future_to_id = {
                    executor.submit(self._process_single_text, text, text_id): text_id
                    for text, text_id in zip(texts, text_ids)
                }

                # 收集结果
                for future in as_completed(future_to_id):
                    text_id = future_to_id[future]
                    try:
                        success = future.result()
                        if success:
                            results["success"] += 1
                        else:
                            results["failed"] += 1
                            results["failed_items"].append(text_id)
                    except Exception as e:
                        logger.error(f"并行处理文本失败: {text_id}, 错误: {e}")
                        results["failed"] += 1
                        results["failed_items"].append(text_id)
        else:
            # 串行处理
            for text, text_id in zip(texts, text_ids):
                if self._process_single_text(text, text_id):
                    results["success"] += 1
                else:
                    results["failed"] += 1
                    results["failed_items"].append(text_id)

        return results

    def process_batch_texts(
        self, texts: List[str], text_ids: List[str]
    ) -> Dict[str, Any]:
        """
        批量处理文本数据

        Args:
            texts: 文本列表
            text_ids: 文本ID列表

        Returns:
            处理结果统计
        """
        if len(texts) != len(text_ids):
            raise ValueError("文本数量和ID数量不匹配")

        results = {"total": len(texts), "success": 0, "failed": 0, "failed_items": []}

        logger.info(f"开始批量处理 {len(texts)} 个文本")

        # 限制批量大小
        max_batch_size = self.config.max_batch_size
        if len(texts) > max_batch_size:
            logger.warning(f"批量大小 {len(texts)} 超过限制 {max_batch_size}，将分批处理")

        # 分批处理
        total_processed = 0
        for i in range(0, len(texts), max_batch_size):
            batch_texts = texts[i : i + max_batch_size]
            batch_ids = text_ids[i : i + max_batch_size]

            logger.info(f"处理批次 {i//max_batch_size + 1}: {len(batch_texts)} 个文本")

            # 处理当前批次
            batch_results = self._process_text_batch(batch_texts, batch_ids)

            # 累计结果
            results["success"] += batch_results["success"]
            results["failed"] += batch_results["failed"]
            results["failed_items"].extend(batch_results["failed_items"])

            total_processed += len(batch_texts)

            # 定期保存索引
            if total_processed % self.config.batch_save_frequency == 0:
                logger.info(f"处理了{total_processed}个项目，保存索引...")
                if self.config.auto_save:
                    self.save_index()

        # 最终保存索引
        if results["success"] > 0 and self.config.auto_save:
            logger.info("批量处理完成，保存最终索引...")
            self.save_index()

        logger.info(f"批量处理完成: 成功{results['success']}, 失败{results['failed']}")
        return results

    def _process_single_image(self, image_base64: str, image_id: str) -> bool:
        """
        处理单个图片（不自动保存索引，用于批量处理）

        Args:
            image_base64: 图片base64编码
            image_id: 图片唯一标识

        Returns:
            是否处理成功
        """
        try:
            response = self.qwen_client.get_image_embedding(image_base64)
            image_vector = np.array(response["embedding"], dtype=np.float32)

            # 验证维度一致性
            if not self._validate_dimension(image_vector, f"图片处理-{image_id}"):
                return False

            # 存储向量（不自动保存索引）
            success = self.vector_store.add_vectors(
                vectors=image_vector.reshape(1, -1), ids=[image_id]
            )

            if success:
                logger.debug(f"成功处理base64图片: {image_id}")
                return True
            else:
                logger.error(f"base64图片处理失败: {image_id}")
                return False

        except Exception as e:
            logger.error(f"处理base64图片失败: {image_id}, 错误: {e}")
            return False

    def _process_image_batch(
        self, images_base64: List[str], image_ids: List[str]
    ) -> Dict[str, Any]:
        """
        内部批量处理图片的实际实现

        Args:
            images_base64: 图片base64列表
            image_ids: 图片ID列表

        Returns:
            处理结果统计
        """
        results = {"success": 0, "failed": 0, "failed_items": []}

        # 使用线程池并行处理
        if self.config.enable_parallel_processing and len(images_base64) > 1:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                # 提交所有任务
                future_to_id = {
                    executor.submit(
                        self._process_single_image, image_base64, image_id
                    ): image_id
                    for image_base64, image_id in zip(images_base64, image_ids)
                }

                # 收集结果
                for future in as_completed(future_to_id):
                    image_id = future_to_id[future]
                    try:
                        success = future.result()
                        if success:
                            results["success"] += 1
                        else:
                            results["failed"] += 1
                            results["failed_items"].append(image_id)
                    except Exception as e:
                        logger.error(f"并行处理图片失败: {image_id}, 错误: {e}")
                        results["failed"] += 1
                        results["failed_items"].append(image_id)
        else:
            # 串行处理
            for image_base64, image_id in zip(images_base64, image_ids):
                if self._process_single_image(image_base64, image_id):
                    results["success"] += 1
                else:
                    results["failed"] += 1
                    results["failed_items"].append(image_id)

        return results

    def process_batch_images(
        self, images_base64: List[str], image_ids: List[str]
    ) -> Dict[str, Any]:
        """
        批量处理图片数据

        Args:
            images_base64: 图片base64列表
            image_ids: 图片ID列表

        Returns:
            处理结果统计
        """
        if len(images_base64) != len(image_ids):
            raise ValueError("图片数量和ID数量不匹配")

        results = {
            "total": len(images_base64),
            "success": 0,
            "failed": 0,
            "failed_items": [],
        }

        logger.info(f"开始批量处理 {len(images_base64)} 个图片")

        # 限制批量大小
        max_batch_size = self.config.max_batch_size
        if len(images_base64) > max_batch_size:
            logger.warning(f"批量大小 {len(images_base64)} 超过限制 {max_batch_size}，将分批处理")

        # 分批处理
        total_processed = 0
        for i in range(0, len(images_base64), max_batch_size):
            batch_images = images_base64[i : i + max_batch_size]
            batch_ids = image_ids[i : i + max_batch_size]

            logger.info(f"处理批次 {i//max_batch_size + 1}: {len(batch_images)} 个图片")

            # 处理当前批次
            batch_results = self._process_image_batch(batch_images, batch_ids)

            # 累计结果
            results["success"] += batch_results["success"]
            results["failed"] += batch_results["failed"]
            results["failed_items"].extend(batch_results["failed_items"])

            total_processed += len(batch_images)

            # 定期保存索引
            if total_processed % self.config.batch_save_frequency == 0:
                logger.info(f"处理了{total_processed}个项目，保存索引...")
                if self.config.auto_save:
                    self.save_index()

        # 最终保存索引
        if results["success"] > 0 and self.config.auto_save:
            logger.info("批量处理完成，保存最终索引...")
            self.save_index()

        logger.info(f"批量处理完成: 成功{results['success']}, 失败{results['failed']}")
        return results

    def search_by_text(
        self, query_text: str, top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        基于文本查询搜索相似向量

        Args:
            query_text: 查询文本
            top_k: 返回结果数量

        Returns:
            搜索结果列表
        """
        try:
            if top_k is None:
                top_k = self.config.default_top_k
            logger.info(f"文本搜索: {query_text}, top_k: {top_k}")

            # 生成查询向量
            response = self.qwen_client.get_text_embedding(query_text)
            query_vector = np.array(response["embedding"], dtype=np.float32)

            # 搜索相似向量
            distances, indices, ids = self.vector_store.search(
                query_vectors=query_vector.reshape(1, -1), k=top_k
            )

            # 整理搜索结果
            results = []
            for i, (distance, _, vector_id) in enumerate(zip(distances, indices, ids)):
                if vector_id:  # 确保ID有效
                    similarity_score = (
                        float(1.0 / (1.0 + distance)) if distance > 0 else 1.0
                    )

                    results.append(
                        {
                            "vector_id": vector_id,
                            "similarity_score": similarity_score,
                            "distance": float(distance),
                            "rank": i + 1,
                        }
                    )

            logger.info(f"文本搜索完成，返回 {len(results)} 个结果")
            return results

        except Exception as e:
            logger.error(f"文本搜索失败: {e}")
            return []

    def search_by_image(
        self, image_base64: str, top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        基于图片查询搜索相似向量

        Args:
            image_base64: 查询图片base64
            top_k: 返回结果数量

        Returns:
            搜索结果列表
        """
        try:
            if top_k is None:
                top_k = self.config.default_top_k
            logger.info(f"图片搜索, top_k: {top_k}")

            # 生成查询向量
            response = self.qwen_client.get_image_embedding(image_base64)
            query_vector = np.array(response["embedding"], dtype=np.float32)

            # 搜索相似向量
            distances, indices, ids = self.vector_store.search(
                query_vectors=query_vector.reshape(1, -1), k=top_k
            )

            # 整理搜索结果
            results = []
            for i, (distance, _, vector_id) in enumerate(zip(distances, indices, ids)):
                if vector_id:
                    similarity_score = (
                        float(1.0 / (1.0 + distance)) if distance > 0 else 1.0
                    )

                    results.append(
                        {
                            "vector_id": vector_id,
                            "similarity_score": similarity_score,
                            "distance": float(distance),
                            "rank": i + 1,
                        }
                    )

            logger.info(f"图片搜索完成，返回 {len(results)} 个结果")
            return results

        except Exception as e:
            logger.error(f"图片搜索失败: {e}")
            return []

    def save_index(self) -> bool:
        """
        保存向量索引到磁盘

        Returns:
            是否保存成功
        """
        try:
            success = self.vector_store.save_index(self.index_save_path)
            if success:
                logger.info(f"索引保存成功: {self.index_save_path}")
            else:
                logger.error(f"索引保存失败: {self.index_save_path}")
            return success
        except Exception as e:
            logger.error(f"保存索引失败: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取处理统计信息

        Returns:
            统计信息字典
        """
        vector_stats = self.vector_store.get_statistics()

        return {
            "total_vectors": vector_stats.get("total_vectors", 0),
            "index_dimension": vector_stats.get("dimension", 0),
            "index_type": vector_stats.get("index_type", ""),
            "embedding_dimension": self.embedding_dimension,
            "index_path": self.index_save_path,
            "id_mapping_size": vector_stats.get("id_mapping_size", 0),
            "config": {
                "batch_save_frequency": self.config.batch_save_frequency,
                "max_batch_size": self.config.max_batch_size,
                "parallel_processing": self.config.enable_parallel_processing,
                "max_workers": self.config.max_workers,
                "auto_save": self.config.auto_save,
            },
        }

    def rebuild_index(self, remove_ids: Optional[List[str]] = None) -> bool:
        """
        重建向量索引

        Args:
            remove_ids: 要删除的ID列表

        Returns:
            是否重建成功
        """
        try:
            logger.info(f"开始重建索引，删除ID数量: {len(remove_ids) if remove_ids else 0}")
            success = self.vector_store.rebuild_index(remove_ids)
            if success:
                logger.info("索引重建成功，保存到磁盘...")
                self.save_index()
            return success
        except Exception as e:
            logger.error(f"重建索引失败: {e}")
            return False

    def remove_vectors(self, vector_ids: List[str]) -> bool:
        """
        删除指定的向量

        Args:
            vector_ids: 要删除的向量ID列表

        Returns:
            是否删除成功
        """
        try:
            if vector_ids:
                logger.info(f"删除 {len(vector_ids)} 个向量")
                return self.rebuild_index(vector_ids)
            else:
                logger.warning("没有指定要删除的向量ID")
                return True

        except Exception as e:
            logger.error(f"删除向量失败, 错误: {e}")
            return False
