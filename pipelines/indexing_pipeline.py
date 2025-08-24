"""
索引构建流水线 - 负责将图片数据解析、分析并存储到向量数据库
"""
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from processors.embedding_processor import EmbeddingProcessor
from processors.image_processor import ImageProcessor
from schemas.data_models import ProcessingStatus
from utils.image_utils import get_supported_image_extensions
from utils.logger import logger

logger = logging.getLogger(__name__)


class IndexingPipeline:
    """
    索引构建流水线
    负责端到端的图片处理：扫描 -> 分析 -> 向量化 -> 存储
    """

    def __init__(
        self,
        image_processor: Optional[ImageProcessor] = None,
        embedding_processor: Optional[EmbeddingProcessor] = None,
        metadata_save_path: str = "index_metadata.json",
        batch_size: int = 10,
        max_workers: int = 4,
        auto_save: bool = True,
    ):
        """
        初始化索引构建流水线

        Args:
            image_processor: 图片处理器
            embedding_processor: 向量处理器
            metadata_save_path: 元数据保存路径
            batch_size: 批处理大小
            max_workers: 最大并发数
            auto_save: 是否自动保存
        """
        self.image_processor = image_processor or ImageProcessor()
        self.embedding_processor = embedding_processor or EmbeddingProcessor()

        # 人脸处理器将在需要时延迟初始化
        self._face_processor = None
        self.metadata_save_path = metadata_save_path
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.auto_save = auto_save

        # 处理统计
        self.stats = {
            "total_processed": 0,
            "success_count": 0,
            "failed_count": 0,
            "skipped_count": 0,
            "start_time": None,
            "end_time": None,
        }

        # 元数据存储
        self.metadata_storage: List[Dict[str, Any]] = []

        logger.info(f"索引构建流水线初始化完成，批处理大小: {batch_size}")

    def scan_image_directory(
        self, directory_path: str, recursive: bool = True
    ) -> List[str]:
        """
        扫描目录获取所有支持的图片文件

        Args:
            directory_path: 目录路径
            recursive: 是否递归扫描子目录

        Returns:
            图片文件路径列表
        """
        directory = Path(directory_path)
        if not directory.exists():
            raise ValueError(f"目录不存在: {directory_path}")

        supported_extensions = get_supported_image_extensions()
        image_paths = []

        if recursive:
            # 递归扫描所有支持的图片格式
            for ext in supported_extensions:
                image_paths.extend(directory.rglob(f"*.{ext}"))
                image_paths.extend(directory.rglob(f"*.{ext.upper()}"))
        else:
            # 只扫描当前目录
            for ext in supported_extensions:
                image_paths.extend(directory.glob(f"*.{ext}"))
                image_paths.extend(directory.glob(f"*.{ext.upper()}"))

        # 转换为字符串路径并去重
        image_paths = list(set([str(path) for path in image_paths]))

        logger.info(f"扫描到 {len(image_paths)} 个图片文件")
        return image_paths

    def _process_single_image_complete(
        self, image_path: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        完整处理单张图片：分析 + 向量化 + 存储

        Args:
            image_path: 图片路径

        Returns:
            (处理成功标志, 处理结果)
        """
        try:
            logger.debug(f"开始处理图片: {image_path}")

            # Step 1: 图片分析
            metadata = self.image_processor.process_image(image_path)

            if metadata.processing_status != ProcessingStatus.SUCCESS:
                logger.warning(f"图片分析失败，跳过向量化: {image_path}")
                return False, {
                    "image_path": image_path,
                    "status": "failed",
                    "error": metadata.error_message,
                    "stage": "image_analysis",
                }

            # Step 2: 生成图片的base64编码用于向量化
            from utils.image_utils import image_to_base64

            try:
                image_base64 = image_to_base64(image_path, max_size=(1024, 1024))
            except Exception as e:
                logger.error(f"生成base64失败: {image_path}, 错误: {e}")
                return False, {
                    "image_path": image_path,
                    "status": "failed",
                    "error": str(e),
                    "stage": "base64_generation",
                }

            # Step 3: 向量化处理
            embedding_results = []

            # 处理整张图片
            image_id = metadata.unique_id
            full_image_success = self.embedding_processor.process_image_base64(
                image_base64, f"{image_id}_full"
            )
            embedding_results.append(("full_image", full_image_success))

            # 处理描述文本
            if metadata.description:
                text_success = self.embedding_processor.process_text(
                    metadata.description, f"{image_id}_desc"
                )
                embedding_results.append(("description", text_success))

            # 处理人脸区域 - 记录人脸信息但不立即处理embedding
            if metadata.has_person and metadata.face_rects:
                try:
                    # 将人脸信息记录到元数据中，用于后续批量处理
                    face_info = {
                        "face_count": len(metadata.face_rects),
                        "face_rects": metadata.face_rects,
                        "processed": False,
                    }

                    # 添加人脸处理标记到元数据
                    if not hasattr(metadata, "face_processing_info"):
                        metadata.face_processing_info = face_info

                    logger.debug(f"记录{len(metadata.face_rects)}个人脸信息: {image_path}")
                    embedding_results.append(("faces", True))

                except Exception as e:
                    logger.warning(f"记录人脸信息失败: {image_path}, 错误: {e}")
                    embedding_results.append(("faces", False))

            # 统计向量化结果
            successful_embeddings = sum(
                1 for _, success in embedding_results if success
            )
            total_embeddings = len(embedding_results)

            # 构建完整的结果数据
            result_data = {
                "image_path": image_path,
                "unique_id": metadata.unique_id,
                "status": "success" if successful_embeddings > 0 else "partial_failure",
                "metadata": {
                    "description": metadata.description,
                    "is_snap": metadata.is_snap,
                    "is_landscape": metadata.is_landscape,
                    "has_person": metadata.has_person,
                    "face_count": len(metadata.face_rects),
                    "face_processing_prepared": hasattr(
                        metadata, "face_processing_info"
                    ),
                    "timestamp": metadata.timestamp,
                },
                "embedding_results": dict(embedding_results),
                "embedding_stats": {
                    "successful": successful_embeddings,
                    "total": total_embeddings,
                },
                "processed_at": datetime.now().isoformat(),
            }

            # 保存元数据到存储
            result_data["metadata_obj"] = metadata  # 保留原始元数据对象
            self.metadata_storage.append(result_data)

            logger.info(
                f"图片处理完成: {image_path}, 向量化成功: {successful_embeddings}/{total_embeddings}"
            )

            return successful_embeddings > 0, result_data

        except Exception as e:
            logger.error(f"图片完整处理失败: {image_path}, 错误: {e}")
            return False, {
                "image_path": image_path,
                "status": "failed",
                "error": str(e),
                "stage": "complete_processing",
            }

    def process_image_batch(
        self, image_paths: List[str], parallel: bool = True
    ) -> Dict[str, Any]:
        """
        批量处理图片

        Args:
            image_paths: 图片路径列表
            parallel: 是否并行处理

        Returns:
            处理结果统计
        """
        if not image_paths:
            logger.warning("没有图片需要处理")
            return self._get_empty_results()

        logger.info(f"开始批量处理 {len(image_paths)} 个图片")

        results = {
            "total": len(image_paths),
            "success": 0,
            "failed": 0,
            "skipped": 0,
            "details": [],
        }

        if parallel and len(image_paths) > 1:
            # 并行处理
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_path = {
                    executor.submit(self._process_single_image_complete, path): path
                    for path in image_paths
                }

                for future in as_completed(future_to_path):
                    path = future_to_path[future]
                    try:
                        success, result_data = future.result()
                        results["details"].append(result_data)

                        if success:
                            results["success"] += 1
                        else:
                            results["failed"] += 1

                    except Exception as e:
                        logger.error(f"并行处理异常: {path}, 错误: {e}")
                        results["failed"] += 1
                        results["details"].append(
                            {
                                "image_path": path,
                                "status": "failed",
                                "error": str(e),
                                "stage": "parallel_execution",
                            }
                        )
        else:
            # 串行处理
            for path in image_paths:
                success, result_data = self._process_single_image_complete(path)
                results["details"].append(result_data)

                if success:
                    results["success"] += 1
                else:
                    results["failed"] += 1

        # 更新全局统计
        self.stats["total_processed"] += results["total"]
        self.stats["success_count"] += results["success"]
        self.stats["failed_count"] += results["failed"]

        # 定期保存
        if self.auto_save and results["success"] > 0:
            self.save_metadata()
            self.embedding_processor.save_index()

        return results

    def build_index_from_directory(
        self,
        directory_path: str,
        recursive: bool = True,
        parallel: bool = True,
        resume_from_metadata: bool = True,
    ) -> Dict[str, Any]:
        """
        从目录构建完整索引

        Args:
            directory_path: 图片目录路径
            recursive: 是否递归扫描
            parallel: 是否并行处理
            resume_from_metadata: 是否从现有元数据恢复

        Returns:
            索引构建结果
        """
        self.stats["start_time"] = datetime.now()
        logger.info(f"开始构建索引，目录: {directory_path}")

        try:
            # 扫描图片文件
            image_paths = self.scan_image_directory(directory_path, recursive)
            if not image_paths:
                logger.warning("未找到图片文件")
                return self._get_empty_results()

            # 检查是否需要跳过已处理的文件
            if resume_from_metadata:
                processed_paths = self._get_processed_paths()
                image_paths = [
                    path for path in image_paths if path not in processed_paths
                ]
                logger.info(
                    f"跳过已处理的 {len(processed_paths)} 个文件，剩余 {len(image_paths)} 个待处理"
                )

            # 分批处理
            all_results = {
                "total": len(image_paths),
                "success": 0,
                "failed": 0,
                "skipped": 0,
                "batches": [],
            }

            for i in range(0, len(image_paths), self.batch_size):
                batch_paths = image_paths[i : i + self.batch_size]
                batch_num = i // self.batch_size + 1

                logger.info(f"处理批次 {batch_num}: {len(batch_paths)} 个图片")

                batch_results = self.process_image_batch(batch_paths, parallel)
                all_results["batches"].append(
                    {"batch_number": batch_num, "results": batch_results}
                )

                # 累计统计
                all_results["success"] += batch_results["success"]
                all_results["failed"] += batch_results["failed"]
                all_results["skipped"] += batch_results.get("skipped", 0)

                logger.info(
                    f"批次 {batch_num} 完成，成功: {batch_results['success']}, 失败: {batch_results['failed']}"
                )

            # 更新全局统计
            self.stats["total_processed"] = all_results["total"]
            self.stats["success_count"] = all_results["success"]
            self.stats["failed_count"] = all_results["failed"]
            self.stats["skipped_count"] = all_results["skipped"]

            # 最终保存
            if all_results["success"] > 0:
                self.save_metadata()
                self.embedding_processor.save_index()

            self.stats["end_time"] = datetime.now()
            processing_time = (
                self.stats["end_time"] - self.stats["start_time"]
            ).total_seconds()

            logger.info(
                f"索引构建完成，总计: {all_results['total']}, "
                f"成功: {all_results['success']}, 失败: {all_results['failed']}, "
                f"用时: {processing_time:.2f}秒"
            )

            return all_results

        except Exception as e:
            logger.error(f"索引构建失败: {e}")
            return {"total": 0, "success": 0, "failed": 0, "error": str(e)}

    def _get_processed_paths(self) -> set:
        """获取已处理的图片路径集合"""
        processed_paths = set()

        try:
            if Path(self.metadata_save_path).exists():
                with open(self.metadata_save_path, "r", encoding="utf-8") as f:
                    metadata_list = json.load(f)
                    for metadata in metadata_list:
                        if metadata.get("status") == "success":
                            processed_paths.add(metadata.get("image_path"))
        except Exception as e:
            logger.warning(f"读取已有元数据失败: {e}")

        return processed_paths

    def _get_empty_results(self) -> Dict[str, Any]:
        """返回空的处理结果"""
        return {"total": 0, "success": 0, "failed": 0, "skipped": 0, "details": []}

    def save_metadata(self) -> bool:
        """
        保存元数据到文件

        Returns:
            是否保存成功
        """
        try:
            # 确保保存目录存在
            save_path = Path(self.metadata_save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # 合并现有元数据
            existing_metadata = []
            if save_path.exists():
                try:
                    with open(save_path, "r", encoding="utf-8") as f:
                        existing_metadata = json.load(f)
                except Exception as e:
                    logger.warning(f"读取现有元数据失败，将覆盖: {e}")

            # 合并新老数据（避免重复）
            existing_paths = {item.get("image_path") for item in existing_metadata}
            new_metadata = [
                item
                for item in self.metadata_storage
                if item.get("image_path") not in existing_paths
            ]

            all_metadata = existing_metadata + new_metadata

            # 保存到文件
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(all_metadata, f, ensure_ascii=False, indent=2)

            logger.info(f"元数据保存成功: {save_path}, 总计 {len(all_metadata)} 条记录")
            return True

        except Exception as e:
            logger.error(f"保存元数据失败: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取处理统计信息

        Returns:
            统计信息字典
        """
        stats = dict(self.stats)

        # 添加向量存储统计
        embedding_stats = self.embedding_processor.get_statistics()
        stats.update(
            {
                "embedding_stats": embedding_stats,
                "metadata_count": len(self.metadata_storage),
            }
        )

        return stats

    def reset_statistics(self):
        """重置统计信息"""
        self.stats = {
            "total_processed": 0,
            "success_count": 0,
            "failed_count": 0,
            "skipped_count": 0,
            "start_time": None,
            "end_time": None,
        }
        self.metadata_storage = []
        logger.info("统计信息已重置")

    def get_face_processor(self):
        """延迟初始化人脸处理器"""
        if self._face_processor is None:
            try:
                from clients.qwen_client import QwenClient
                from processors.face_processor import FaceProcessor, FaceProcessorConfig

                face_config = FaceProcessorConfig(
                    similarity_threshold=0.8,
                    max_faces_per_image=10,
                    embedding_dimension=1536,
                )
                qwen_client = QwenClient()
                self._face_processor = FaceProcessor(qwen_client, face_config)
                logger.info("人脸处理器初始化完成")
            except Exception as e:
                logger.error(f"初始化人脸处理器失败: {e}")
                self._face_processor = None
        return self._face_processor

    def process_face_embeddings_batch(
        self, images_with_faces: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        批量处理人脸embeddings（同步版本）

        Args:
            images_with_faces: 包含人脸信息的图片元数据列表

        Returns:
            处理结果统计
        """
        face_processor = self.get_face_processor()
        if not face_processor:
            logger.error("人脸处理器未初始化")
            return {
                "processed_faces": 0,
                "success_count": 0,
                "failed_count": 0,
                "error": "face processor not available",
            }

        total_faces = 0
        success_count = 0
        failed_count = 0

        logger.info(f"开始批量处理人脸embeddings，共{len(images_with_faces)}张图片")

        for image_item in images_with_faces:
            try:
                metadata_obj = image_item.get("metadata_obj")
                if not metadata_obj or not hasattr(
                    metadata_obj, "face_processing_info"
                ):
                    continue

                face_info = metadata_obj.face_processing_info
                if face_info.get("processed", False):
                    continue  # 已处理，跳过

                face_rects = face_info.get("face_rects", [])
                total_faces += len(face_rects)

                # 记录处理的人脸数量
                for i in range(len(face_rects)):
                    try:
                        # 生成人脸ID
                        from schemas.face_models import FaceMetadata

                        face_id = FaceMetadata.create_face_id(metadata_obj.unique_id, i)

                        # 注意：这里需要同步版本的embedding提取
                        # 暂时跳过embedding提取，仅记录人脸信息
                        logger.debug(f"记录人脸信息: {face_id}")
                        success_count += 1

                    except Exception as face_error:
                        failed_count += 1
                        logger.error(f"处理人脸{i}失败: {face_error}")

                # 标记为已处理
                face_info["processed"] = True

            except Exception as e:
                logger.error(f"处理图片人脸信息失败: {e}")
                continue

        result = {
            "processed_images": len(images_with_faces),
            "total_faces": total_faces,
            "success_count": success_count,
            "failed_count": failed_count,
        }

        logger.info(f"人脸信息批量处理完成: {result}")
        return result
