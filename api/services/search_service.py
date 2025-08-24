"""
搜索业务服务
"""
import time
from typing import Any, Dict, List, Optional

from pipelines.retrieval_pipeline import RetrievalPipeline
from processors.face_processor import FaceProcessor
from schemas.data_models import SearchResult
from utils.logger import setup_logger

logger = setup_logger(__name__)


class SearchService:
    """搜索业务服务"""

    def __init__(
        self, retrieval_pipeline: RetrievalPipeline, face_processor: FaceProcessor
    ):
        self.retrieval_pipeline = retrieval_pipeline
        self.face_processor = face_processor

        logger.info("搜索服务初始化完成")

    async def search_by_text(
        self,
        query: str,
        limit: int = 10,
        similarity_threshold: float = 0.5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        文本搜索图片

        Args:
            query: 搜索查询文本
            limit: 返回结果数量
            similarity_threshold: 相似度阈值
            filters: 搜索过滤条件

        Returns:
            搜索结果列表
        """
        try:
            logger.info(f"执行文本搜索: '{query}', limit={limit}")

            # 调用检索管道进行文本搜索
            search_results = await self.retrieval_pipeline.search_by_text(
                query=query, top_k=limit, similarity_threshold=similarity_threshold
            )

            # 应用额外的过滤条件
            if filters:
                search_results = self._apply_filters(search_results, filters)

            logger.info(f"文本搜索完成: 返回{len(search_results)}个结果")
            return search_results

        except Exception as e:
            logger.error(f"文本搜索失败: {e}")
            raise

    async def search_by_image(
        self,
        image_path: str,
        limit: int = 10,
        similarity_threshold: float = 0.5,
        search_faces: bool = False,
    ) -> List[SearchResult]:
        """
        图像搜索

        Args:
            image_path: 查询图片路径
            limit: 返回结果数量
            similarity_threshold: 相似度阈值
            search_faces: 是否同时搜索人脸

        Returns:
            搜索结果列表
        """
        try:
            logger.info(f"执行图像搜索: {image_path}, search_faces={search_faces}")

            if search_faces:
                # 同时进行图像和人脸搜索
                return await self._search_image_and_faces(
                    image_path, limit, similarity_threshold
                )
            else:
                # 仅进行图像搜索
                search_results = await self.retrieval_pipeline.search_by_image(
                    query_image_path=image_path,
                    top_k=limit,
                    similarity_threshold=similarity_threshold,
                )

                logger.info(f"图像搜索完成: 返回{len(search_results)}个结果")
                return search_results

        except Exception as e:
            logger.error(f"图像搜索失败: {e}")
            raise

    async def _search_image_and_faces(
        self, image_path: str, limit: int, similarity_threshold: float
    ) -> List[SearchResult]:
        """
        同时进行图像和人脸搜索，合并结果
        """
        try:
            # 并行执行图像搜索和人脸搜索
            image_results_task = self.retrieval_pipeline.search_by_image(
                query_image_path=image_path,
                top_k=limit,
                similarity_threshold=similarity_threshold,
            )

            face_results_task = self.face_processor.search_faces_by_image(
                query_image_path=image_path,
                top_k=limit // 2,  # 人脸搜索返回一半的结果
                similarity_threshold=max(0.8, similarity_threshold),  # 人脸搜索使用更高的阈值
            )

            # 等待两个搜索完成
            import asyncio

            image_results, face_results = await asyncio.gather(
                image_results_task, face_results_task, return_exceptions=True
            )

            # 处理搜索结果
            combined_results = []

            if not isinstance(image_results, Exception):
                combined_results.extend(image_results)

            if not isinstance(face_results, Exception) and face_results:
                # 将人脸搜索结果转换为图像搜索结果格式
                for face_result in face_results:
                    # 检查是否已经存在相同的图片
                    existing_paths = [r.metadata.path for r in combined_results]
                    if face_result.face_metadata.image_path not in existing_paths:
                        # 创建伪装的SearchResult (这里需要根据实际的SearchResult结构调整)
                        # TODO: 需要实际的SearchResult创建逻辑
                        pass

            # 按相似度排序并限制结果数量
            combined_results.sort(key=lambda x: x.similarity_score, reverse=True)
            return combined_results[:limit]

        except Exception as e:
            logger.error(f"图像和人脸联合搜索失败: {e}")
            # 降级到仅图像搜索
            return await self.retrieval_pipeline.search_by_image(
                query_image_path=image_path,
                top_k=limit,
                similarity_threshold=similarity_threshold,
            )

    def _apply_filters(
        self, search_results: List[SearchResult], filters: Dict[str, Any]
    ) -> List[SearchResult]:
        """
        应用搜索过滤条件

        Args:
            search_results: 原始搜索结果
            filters: 过滤条件

        Returns:
            过滤后的搜索结果
        """
        try:
            filtered_results = []

            for result in search_results:
                metadata = result.metadata

                # 应用各种过滤条件
                if filters.get("is_snap") is not None:
                    if metadata.is_snap != filters["is_snap"]:
                        continue

                if filters.get("is_landscape") is not None:
                    if metadata.is_landscape != filters["is_landscape"]:
                        continue

                if filters.get("has_person") is not None:
                    if metadata.has_person != filters["has_person"]:
                        continue

                if filters.get("min_faces"):
                    if len(metadata.face_rects) < filters["min_faces"]:
                        continue

                if filters.get("max_faces"):
                    if len(metadata.face_rects) > filters["max_faces"]:
                        continue

                # 时间范围过滤
                if filters.get("date_range"):
                    # TODO: 实现时间范围过滤逻辑
                    pass

                # 关键词过滤
                if filters.get("keywords"):
                    keywords = filters["keywords"]
                    if isinstance(keywords, str):
                        keywords = [keywords]

                    description_lower = metadata.description.lower()
                    if not any(
                        keyword.lower() in description_lower for keyword in keywords
                    ):
                        continue

                filtered_results.append(result)

            logger.info(f"过滤结果: {len(search_results)} -> {len(filtered_results)}")
            return filtered_results

        except Exception as e:
            logger.error(f"应用过滤条件失败: {e}")
            return search_results  # 过滤失败时返回原始结果

    async def get_search_suggestions(self, partial_query: str) -> List[str]:
        """
        获取搜索建议

        Args:
            partial_query: 部分查询文本

        Returns:
            搜索建议列表
        """
        try:
            # TODO: 实现基于历史搜索和内容的搜索建议
            # 这里可以基于：
            # 1. 历史搜索记录
            # 2. 图片描述中的高频词汇
            # 3. 预定义的常用搜索词

            suggestions = []

            # 简单的建议逻辑（示例）
            common_suggestions = [
                "风景照",
                "人物照",
                "截图",
                "游戏截图",
                "自然风光",
                "城市风景",
                "人像",
                "合影",
                "动物",
                "建筑",
            ]

            partial_lower = partial_query.lower()
            for suggestion in common_suggestions:
                if partial_lower in suggestion.lower() or suggestion.lower().startswith(
                    partial_lower
                ):
                    suggestions.append(suggestion)

            return suggestions[:5]  # 返回前5个建议

        except Exception as e:
            logger.error(f"获取搜索建议失败: {e}")
            return []
