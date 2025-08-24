"""
人脸识别业务服务
"""
import time
from typing import Any, Dict, List, Optional

from api.models.responses import DetectedFace
from processors.face_processor import FaceProcessor
from schemas.face_models import FaceComparisonResult, FaceSearchResult
from utils.logger import setup_logger

logger = setup_logger(__name__)


class FaceDetectionResult:
    """人脸检测结果包装类"""

    def __init__(self, faces: List[DetectedFace]):
        self.faces = faces


class FaceService:
    """人脸识别业务服务"""

    def __init__(self, face_processor: FaceProcessor):
        self.face_processor = face_processor
        logger.info("人脸服务初始化完成")

    async def detect_faces(
        self, image_path: str, include_embeddings: bool = False
    ) -> FaceDetectionResult:
        """
        检测图片中的人脸

        Args:
            image_path: 图片路径
            include_embeddings: 是否包含人脸特征向量

        Returns:
            人脸检测结果
        """
        try:
            logger.info(f"检测人脸: {image_path}")

            # 首先使用ImageProcessor检测人脸
            from clients.qwen_client import QwenClient
            from processors.image_processor import ImageProcessor

            qwen_client = QwenClient()
            image_processor = ImageProcessor(qwen_client)

            # 分析图片获取人脸信息
            image_metadata = await image_processor.analyze_image(image_path)

            detected_faces = []

            if image_metadata.has_person and image_metadata.face_rects:
                logger.info(f"检测到{len(image_metadata.face_rects)}个人脸")

                for i, face_rect in enumerate(image_metadata.face_rects):
                    # 生成人脸ID
                    from schemas.face_models import FaceMetadata

                    face_id = FaceMetadata.create_face_id(
                        f"detect_{hash(image_path)}", i
                    )

                    # 创建检测结果
                    detected_face = DetectedFace(
                        face_id=face_id,
                        bounding_box=list(face_rect),
                        confidence=0.9,  # TODO: 从实际检测结果获取置信度
                    )

                    # 如果需要包含embedding，则提取特征向量
                    if include_embeddings:
                        try:
                            face_embeddings = (
                                await self.face_processor.extract_face_embeddings(
                                    image_path, [face_rect], f"temp_{hash(image_path)}"
                                )
                            )
                            if (
                                face_embeddings
                                and face_embeddings[0].embedding_vector is not None
                            ):
                                detected_face.embedding_vector = face_embeddings[
                                    0
                                ].embedding_vector.tolist()
                        except Exception as e:
                            logger.warning(f"提取人脸embedding失败: {e}")

                    detected_faces.append(detected_face)

            result = FaceDetectionResult(detected_faces)
            logger.info(f"人脸检测完成: 检测到{len(detected_faces)}个人脸")
            return result

        except Exception as e:
            logger.error(f"人脸检测失败: {e}")
            raise

    async def search_faces_by_image(
        self,
        image_path: str,
        limit: int = 10,
        similarity_threshold: float = 0.8,
        similarity_method: str = "cosine_similarity",
    ) -> List[FaceSearchResult]:
        """
        通过图片搜索相似人脸

        Args:
            image_path: 查询图片路径
            limit: 返回结果数量
            similarity_threshold: 相似度阈值
            similarity_method: 相似度计算方法

        Returns:
            人脸搜索结果列表
        """
        try:
            logger.info(f"人脸搜索: {image_path}")

            # 调用人脸处理器进行搜索
            search_results = await self.face_processor.search_faces_by_image(
                query_image_path=image_path,
                top_k=limit,
                similarity_threshold=similarity_threshold,
            )

            logger.info(f"人脸搜索完成: 找到{len(search_results)}个相似人脸")
            return search_results

        except Exception as e:
            logger.error(f"人脸搜索失败: {e}")
            raise

    async def find_similar_faces_by_id(
        self, face_id: str, limit: int = 10, similarity_threshold: float = 0.8
    ) -> List[FaceSearchResult]:
        """
        通过人脸ID查找相似人脸

        Args:
            face_id: 人脸ID
            limit: 返回结果数量
            similarity_threshold: 相似度阈值

        Returns:
            人脸搜索结果列表
        """
        try:
            logger.info(f"通过ID查找相似人脸: {face_id}")

            # 获取人脸元数据
            face_metadata = self.face_processor.face_store.get_face_metadata_by_id(
                face_id
            )
            if not face_metadata or face_metadata.embedding_vector is None:
                logger.warning(f"找不到人脸ID或embedding: {face_id}")
                return []

            # 使用embedding搜索相似人脸
            search_results = await self.face_processor.find_similar_faces(
                query_embedding=face_metadata.embedding_vector,
                top_k=limit + 1,  # +1 因为结果中会包含自己
                similarity_threshold=similarity_threshold,
            )

            # 过滤掉自己
            filtered_results = [
                result
                for result in search_results
                if result.face_metadata.face_id != face_id
            ]

            # 重新排序排名
            for i, result in enumerate(filtered_results):
                result.rank = i + 1

            logger.info(f"相似人脸查找完成: 找到{len(filtered_results)}个相似人脸")
            return filtered_results[:limit]

        except Exception as e:
            logger.error(f"通过ID查找相似人脸失败: {e}")
            raise

    async def compare_faces(
        self, face_id1: str, face_id2: str, similarity_method: str = "cosine_similarity"
    ) -> Optional[FaceComparisonResult]:
        """
        比较两个人脸的相似度

        Args:
            face_id1: 第一个人脸ID
            face_id2: 第二个人脸ID
            similarity_method: 相似度计算方法

        Returns:
            人脸比较结果
        """
        try:
            logger.info(f"比较人脸: {face_id1} vs {face_id2}")

            # 调用人脸处理器进行比较
            comparison_result = self.face_processor.compare_faces(
                face_id1=face_id1,
                face_id2=face_id2,
                similarity_method=similarity_method,
            )

            if comparison_result:
                logger.info(f"人脸比较完成: 相似度={comparison_result.similarity_score:.3f}")
            else:
                logger.warning("人脸比较失败: 找不到指定的人脸ID")

            return comparison_result

        except Exception as e:
            logger.error(f"人脸比较失败: {e}")
            raise

    async def get_face_statistics(self) -> Dict[str, Any]:
        """
        获取人脸识别系统统计信息

        Returns:
            统计信息字典
        """
        try:
            logger.info("获取人脸统计信息")

            # 调用人脸处理器获取统计信息
            stats = self.face_processor.get_face_statistics()

            logger.info("人脸统计信息获取完成")
            return stats

        except Exception as e:
            logger.error(f"获取人脸统计信息失败: {e}")
            raise

    async def get_face_info(self, face_id: str) -> Optional[Dict[str, Any]]:
        """
        获取指定人脸的详细信息

        Args:
            face_id: 人脸ID

        Returns:
            人脸详细信息
        """
        try:
            logger.info(f"获取人脸信息: {face_id}")

            # 从人脸存储中获取人脸元数据
            face_metadata = self.face_processor.face_store.get_face_metadata_by_id(
                face_id
            )

            if face_metadata is None:
                logger.warning(f"找不到人脸ID: {face_id}")
                return None

            # 转换为字典格式
            face_info = face_metadata.to_dict()

            # 添加额外信息
            face_info["has_embedding"] = face_metadata.embedding_vector is not None
            face_info["embedding_dimension"] = (
                len(face_metadata.embedding_vector)
                if face_metadata.embedding_vector is not None
                else 0
            )

            logger.info(f"人脸信息获取完成: {face_id}")
            return face_info

        except Exception as e:
            logger.error(f"获取人脸信息失败: {e}")
            raise

    async def delete_face(self, face_id: str) -> bool:
        """
        删除指定的人脸数据

        Args:
            face_id: 人脸ID

        Returns:
            删除是否成功
        """
        try:
            logger.info(f"删除人脸: {face_id}")

            # 调用人脸存储删除人脸
            success = self.face_processor.face_store.remove_face(face_id)

            if success:
                logger.info(f"人脸删除成功: {face_id}")
            else:
                logger.warning(f"人脸删除失败: {face_id}")

            return success

        except Exception as e:
            logger.error(f"删除人脸失败: {e}")
            raise
