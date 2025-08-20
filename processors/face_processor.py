"""
人脸处理器 - 专门处理人脸检测、向量化和相似度匹配
"""
import asyncio
import logging
import time
from typing import List, Optional, Tuple, Dict, Any
import numpy as np

from clients.qwen_client import QwenClient
from schemas.face_models import (
    FaceMetadata, FaceSearchResult, FaceComparisonResult, 
    FaceBatchProcessingResult, FaceSimilarityMethod
)
from schemas.data_models import ImageMetadata
from vector_store.face_faiss_store import FaceFaissStore
from utils.image_utils import crop_face_from_image
from utils.logger import setup_logger
from utils.retry_utils import with_retry

logger = setup_logger(__name__)


class FaceProcessorConfig:
    """人脸处理器配置"""
    def __init__(
        self,
        similarity_threshold: float = 0.8,
        max_faces_per_image: int = 10,
        face_crop_padding: float = 0.2,
        embedding_dimension: int = 1536,
        enable_batch_processing: bool = True,
        batch_size: int = 10
    ):
        self.similarity_threshold = similarity_threshold
        self.max_faces_per_image = max_faces_per_image
        self.face_crop_padding = face_crop_padding
        self.embedding_dimension = embedding_dimension
        self.enable_batch_processing = enable_batch_processing
        self.batch_size = batch_size


class FaceProcessor:
    """专门的人脸处理器"""
    
    def __init__(
        self, 
        qwen_client: QwenClient, 
        config: Optional[FaceProcessorConfig] = None,
        face_store: Optional[FaceFaissStore] = None
    ):
        """
        初始化人脸处理器
        
        Args:
            qwen_client: Qwen客户端
            config: 人脸处理器配置
            face_store: 人脸向量存储（可选）
        """
        self.qwen_client = qwen_client
        self.config = config or FaceProcessorConfig()
        
        # 初始化人脸向量存储
        self.face_store = face_store or FaceFaissStore(
            dimension=self.config.embedding_dimension,
            index_type="IndexFlatIP"
        )
        
        logger.info(f"人脸处理器初始化完成: embedding_dim={self.config.embedding_dimension}")
    
    async def extract_face_embeddings(
        self, 
        image_path: str, 
        face_rects: List[Tuple[int, int, int, int]],
        image_id: Optional[str] = None
    ) -> List[FaceMetadata]:
        """
        从图片中提取人脸embeddings
        
        Args:
            image_path: 图片路径
            face_rects: 人脸框坐标列表 [(x, y, w, h), ...]
            image_id: 图片ID
            
        Returns:
            人脸元数据列表
        """
        try:
            if not face_rects:
                logger.info("没有检测到人脸，跳过embedding提取")
                return []
            
            # 限制人脸数量
            if len(face_rects) > self.config.max_faces_per_image:
                logger.warning(f"人脸数量超过限制，仅处理前{self.config.max_faces_per_image}个")
                face_rects = face_rects[:self.config.max_faces_per_image]
            
            face_metadata_list = []
            
            for i, face_rect in enumerate(face_rects):
                try:
                    # 生成人脸ID
                    face_id = FaceMetadata.create_face_id(
                        image_id or f"img_{hash(image_path)}", i
                    )
                    
                    # 裁剪人脸区域
                    face_base64 = crop_face_from_image(
                        image_path, face_rect, self.config.face_crop_padding
                    )
                    
                    # 提取人脸embedding
                    embedding = await self._extract_single_face_embedding(face_base64)
                    
                    # 创建人脸元数据
                    face_metadata = FaceMetadata(
                        face_id=face_id,
                        image_id=image_id or f"img_{hash(image_path)}",
                        image_path=image_path,
                        face_rect=face_rect,
                        confidence_score=0.9,  # TODO: 从检测结果获取实际置信度
                        embedding_vector=embedding,
                        embedding_dimension=len(embedding)
                    )
                    
                    face_metadata_list.append(face_metadata)
                    
                    logger.debug(f"人脸embedding提取成功: {face_id}")
                    
                except Exception as e:
                    logger.error(f"提取人脸{i}的embedding失败: {e}")
                    continue
            
            logger.info(f"成功提取{len(face_metadata_list)}/{len(face_rects)}个人脸embedding")
            return face_metadata_list
            
        except Exception as e:
            logger.error(f"提取人脸embeddings失败: {e}")
            raise
    
    @with_retry(max_retries=3, base_delay=1.0)
    async def _extract_single_face_embedding(self, face_base64: str) -> np.ndarray:
        """
        提取单个人脸的embedding向量
        
        Args:
            face_base64: 人脸图片的base64编码
            
        Returns:
            人脸embedding向量
        """
        try:
            # 使用Qwen多模态模型提取人脸特征
            prompt = "请分析这张人脸图片，提取用于人脸识别的特征向量。注重面部轮廓、眼睛、鼻子、嘴巴等关键特征。"
            
            # 调用Qwen的图像embedding接口
            embedding = await self.qwen_client.get_image_embedding(
                image_base64=face_base64,
                instruction=prompt
            )
            
            if embedding is None or len(embedding) == 0:
                raise ValueError("获取的embedding为空")
            
            # 确保embedding是numpy数组
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding)
            
            # L2归一化
            embedding = embedding / np.linalg.norm(embedding)
            
            logger.debug(f"成功提取人脸embedding: 维度={len(embedding)}")
            return embedding.astype('float32')
            
        except Exception as e:
            logger.error(f"提取人脸embedding失败: {e}")
            raise
    
    async def process_image_faces(
        self, 
        image_metadata: ImageMetadata
    ) -> List[FaceMetadata]:
        """
        处理图片中的所有人脸
        
        Args:
            image_metadata: 图片元数据
            
        Returns:
            人脸元数据列表
        """
        try:
            if not image_metadata.has_person or not image_metadata.face_rects:
                logger.info(f"图片{image_metadata.unique_id}没有人脸，跳过处理")
                return []
            
            # 提取人脸embeddings
            face_metadata_list = await self.extract_face_embeddings(
                image_metadata.path,
                image_metadata.face_rects,
                image_metadata.unique_id
            )
            
            # 将人脸添加到向量存储
            for face_metadata in face_metadata_list:
                try:
                    self.face_store.add_face_embedding(
                        face_metadata.embedding_vector,
                        face_metadata
                    )
                    logger.debug(f"人脸已添加到向量存储: {face_metadata.face_id}")
                except Exception as e:
                    logger.error(f"添加人脸到向量存储失败: {e}")
            
            return face_metadata_list
            
        except Exception as e:
            logger.error(f"处理图片人脸失败: {e}")
            raise
    
    async def find_similar_faces(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 10, 
        similarity_threshold: Optional[float] = None,
        similarity_method: str = FaceSimilarityMethod.COSINE
    ) -> List[FaceSearchResult]:
        """
        查找相似人脸
        
        Args:
            query_embedding: 查询人脸的embedding
            top_k: 返回结果数量
            similarity_threshold: 相似度阈值
            similarity_method: 相似度计算方法
            
        Returns:
            人脸搜索结果列表
        """
        try:
            # 使用配置的阈值
            threshold = similarity_threshold or self.config.similarity_threshold
            
            # 搜索相似人脸
            search_results = self.face_store.search_similar_faces(
                query_embedding, top_k, similarity_method
            )
            
            # 应用相似度阈值过滤
            filtered_results = []
            for result in search_results:
                if result.similarity_score >= threshold:
                    filtered_results.append(result)
                else:
                    logger.debug(f"人脸相似度低于阈值: {result.similarity_score} < {threshold}")
            
            logger.info(f"找到{len(filtered_results)}个满足阈值的相似人脸")
            return filtered_results
            
        except Exception as e:
            logger.error(f"查找相似人脸失败: {e}")
            raise
    
    async def search_faces_by_image(
        self, 
        query_image_path: str,
        top_k: int = 10,
        similarity_threshold: Optional[float] = None
    ) -> List[FaceSearchResult]:
        """
        通过图片搜索相似人脸
        
        Args:
            query_image_path: 查询图片路径
            top_k: 返回结果数量
            similarity_threshold: 相似度阈值
            
        Returns:
            人脸搜索结果列表
        """
        try:
            # 首先检测查询图片中的人脸
            from processors.image_processor import ImageProcessor
            image_processor = ImageProcessor(self.qwen_client)
            
            # 分析图片获取人脸信息
            query_metadata = await image_processor.analyze_image(query_image_path)
            
            if not query_metadata.has_person or not query_metadata.face_rects:
                logger.warning("查询图片中没有检测到人脸")
                return []
            
            # 提取查询图片中所有人脸的embeddings
            query_face_embeddings = await self.extract_face_embeddings(
                query_image_path,
                query_metadata.face_rects,
                f"query_{hash(query_image_path)}"
            )
            
            # 对每个查询人脸进行搜索
            all_results = []
            for query_face in query_face_embeddings:
                similar_faces = await self.find_similar_faces(
                    query_face.embedding_vector,
                    top_k,
                    similarity_threshold
                )
                all_results.extend(similar_faces)
            
            # 去重并按相似度排序
            unique_results = {}
            for result in all_results:
                face_id = result.face_metadata.face_id
                if (face_id not in unique_results or 
                    result.similarity_score > unique_results[face_id].similarity_score):
                    unique_results[face_id] = result
            
            # 重新排序和排名
            sorted_results = sorted(
                unique_results.values(), 
                key=lambda x: x.similarity_score, 
                reverse=True
            )[:top_k]
            
            # 更新排名
            for i, result in enumerate(sorted_results):
                result.rank = i + 1
            
            logger.info(f"通过图片搜索到{len(sorted_results)}个相似人脸")
            return sorted_results
            
        except Exception as e:
            logger.error(f"通过图片搜索人脸失败: {e}")
            raise
    
    def compare_faces(
        self, 
        face_id1: str, 
        face_id2: str,
        similarity_method: str = FaceSimilarityMethod.COSINE
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
            similarity_score = self.face_store.compare_faces(
                face_id1, face_id2, similarity_method
            )
            
            if similarity_score is None:
                return None
            
            # 计算距离（从相似度反推）
            if similarity_method == FaceSimilarityMethod.COSINE:
                distance = 2.0 * (1.0 - similarity_score)
            elif similarity_method == FaceSimilarityMethod.EUCLIDEAN:
                distance = -2.0 * np.log(max(similarity_score, 1e-10))
            else:
                distance = 1.0 - similarity_score
            
            comparison_result = FaceComparisonResult(
                face1_id=face_id1,
                face2_id=face_id2,
                similarity_score=similarity_score,
                distance=distance,
                is_same_person=similarity_score >= self.config.similarity_threshold,
                confidence_threshold=self.config.similarity_threshold,
                comparison_method=similarity_method
            )
            
            logger.info(f"人脸比较完成: {face_id1} vs {face_id2}, "
                       f"相似度={similarity_score:.3f}")
            
            return comparison_result
            
        except Exception as e:
            logger.error(f"人脸比较失败: {e}")
            return None
    
    async def batch_process_images(
        self, 
        image_metadata_list: List[ImageMetadata]
    ) -> FaceBatchProcessingResult:
        """
        批量处理图片中的人脸
        
        Args:
            image_metadata_list: 图片元数据列表
            
        Returns:
            批量处理结果
        """
        try:
            start_time = time.time()
            
            result = FaceBatchProcessingResult(
                total_images=len(image_metadata_list),
                total_faces=0,
                successful_extractions=0,
                failed_extractions=0,
                processing_time_seconds=0.0
            )
            
            # 处理每张图片
            for image_metadata in image_metadata_list:
                try:
                    face_metadata_list = await self.process_image_faces(image_metadata)
                    
                    if face_metadata_list:
                        result.successful_extractions += 1
                        for face_metadata in face_metadata_list:
                            result.add_face_metadata(face_metadata)
                    else:
                        # 没有人脸也算成功处理
                        result.successful_extractions += 1
                        
                except Exception as e:
                    error_msg = f"处理图片{image_metadata.path}失败: {e}"
                    result.add_error(error_msg)
                    logger.error(error_msg)
            
            result.processing_time_seconds = time.time() - start_time
            
            logger.info(f"批量人脸处理完成: {result.successful_extractions}/{result.total_images}张图片, "
                       f"{result.total_faces}个人脸, 耗时{result.processing_time_seconds:.2f}秒")
            
            return result
            
        except Exception as e:
            logger.error(f"批量处理人脸失败: {e}")
            raise
    
    def get_face_statistics(self) -> Dict[str, Any]:
        """获取人脸处理统计信息"""
        try:
            face_store_stats = self.face_store.get_statistics()
            
            processor_stats = {
                "config": {
                    "similarity_threshold": self.config.similarity_threshold,
                    "max_faces_per_image": self.config.max_faces_per_image,
                    "embedding_dimension": self.config.embedding_dimension
                },
                "store_stats": face_store_stats
            }
            
            return processor_stats
            
        except Exception as e:
            logger.error(f"获取人脸统计信息失败: {e}")
            return {}
    
    def save_face_index(self):
        """保存人脸索引"""
        try:
            self.face_store.save_index()
            logger.info("人脸索引保存完成")
        except Exception as e:
            logger.error(f"保存人脸索引失败: {e}")
            raise
    
    def load_face_index(self):
        """加载人脸索引"""
        try:
            self.face_store.load_index()
            logger.info("人脸索引加载完成")
        except Exception as e:
            logger.error(f"加载人脸索引失败: {e}")
            raise