"""
检索查询流水线 - 负责处理用户查询并返回相关结果
支持文本查询和图片查询，提供统一的检索接口
"""
import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
import numpy as np

from processors.embedding_processor import EmbeddingProcessor
from clients.qwen_client import QwenClient
from utils.logger import logger
from utils.image_utils import image_to_base64

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    """
    检索查询流水线
    负责处理各种类型的查询请求，并返回排序后的结果
    """
    
    def __init__(
        self,
        embedding_processor: Optional[EmbeddingProcessor] = None,
        metadata_path: str = "index_metadata.json",
        default_top_k: int = 10,
        similarity_threshold: float = 0.3,
        enable_rerank: bool = False
    ):
        """
        初始化检索流水线
        
        Args:
            embedding_processor: 向量处理器
            metadata_path: 元数据文件路径
            default_top_k: 默认返回结果数量
            similarity_threshold: 相似度阈值
            enable_rerank: 是否启用重排序
        """
        self.embedding_processor = embedding_processor or EmbeddingProcessor()
        self.metadata_path = metadata_path
        self.default_top_k = default_top_k
        self.similarity_threshold = similarity_threshold
        self.enable_rerank = enable_rerank
        
        # 加载元数据索引
        self.metadata_index = self._load_metadata_index()
        
        logger.info(f"检索流水线初始化完成，元数据数量: {len(self.metadata_index)}")
    
    def _load_metadata_index(self) -> Dict[str, Dict[str, Any]]:
        """
        加载元数据索引
        
        Returns:
            以unique_id为key的元数据字典
        """
        metadata_index = {}
        
        try:
            metadata_path = Path(self.metadata_path)
            if not metadata_path.exists():
                logger.warning(f"元数据文件不存在: {self.metadata_path}")
                return metadata_index
            
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata_list = json.load(f)
            
            for item in metadata_list:
                if item.get("status") == "success" and item.get("unique_id"):
                    unique_id = item["unique_id"]
                    metadata_index[unique_id] = item
            
            logger.info(f"加载元数据索引成功: {len(metadata_index)} 条记录")
            
        except Exception as e:
            logger.error(f"加载元数据索引失败: {e}")
        
        return metadata_index
    
    def _enhance_results_with_metadata(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        使用元数据增强搜索结果
        
        Args:
            search_results: 原始搜索结果
            
        Returns:
            增强后的搜索结果
        """
        enhanced_results = []
        
        for result in search_results:
            vector_id = result.get("vector_id", "")
            
            # 解析vector_id以获取unique_id
            # vector_id格式: {unique_id}_full, {unique_id}_desc, {unique_id}_face_{i}
            unique_id = self._extract_unique_id_from_vector_id(vector_id)
            
            # 获取对应的元数据
            metadata = self.metadata_index.get(unique_id, {})
            
            enhanced_result = {
                "vector_id": vector_id,
                "unique_id": unique_id,
                "similarity_score": result.get("similarity_score", 0.0),
                "distance": result.get("distance", float('inf')),
                "rank": result.get("rank", 0),
                "match_type": self._determine_match_type(vector_id),
                "image_path": metadata.get("image_path", ""),
                "metadata": metadata.get("metadata", {}),
                "processed_at": metadata.get("processed_at", "")
            }
            
            enhanced_results.append(enhanced_result)
        
        return enhanced_results
    
    def _extract_unique_id_from_vector_id(self, vector_id: str) -> str:
        """
        从vector_id中提取unique_id
        
        Args:
            vector_id: 向量ID
            
        Returns:
            提取的unique_id
        """
        # 移除后缀 _full, _desc, _face_N
        if "_full" in vector_id:
            return vector_id.replace("_full", "")
        elif "_desc" in vector_id:
            return vector_id.replace("_desc", "")
        elif "_face_" in vector_id:
            # 移除 _face_{number} 后缀
            parts = vector_id.split("_face_")
            return parts[0] if parts else vector_id
        elif "_faces" in vector_id:
            return vector_id.replace("_faces", "")
        else:
            return vector_id
    
    def _determine_match_type(self, vector_id: str) -> str:
        """
        判断匹配类型
        
        Args:
            vector_id: 向量ID
            
        Returns:
            匹配类型
        """
        if "_full" in vector_id:
            return "image_full"
        elif "_desc" in vector_id:
            return "description"
        elif "_face_" in vector_id:
            return "face"
        else:
            return "unknown"
    
    def _filter_by_similarity(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        根据相似度阈值过滤结果
        
        Args:
            results: 搜索结果
            
        Returns:
            过滤后的结果
        """
        filtered_results = [
            result for result in results
            if result.get("similarity_score", 0.0) >= self.similarity_threshold
        ]
        
        logger.debug(f"相似度过滤: {len(results)} -> {len(filtered_results)}")
        return filtered_results
    
    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        去重结果（同一张图片的不同匹配类型合并）
        
        Args:
            results: 搜索结果
            
        Returns:
            去重后的结果
        """
        unique_results = {}
        
        for result in results:
            unique_id = result.get("unique_id", "")
            if not unique_id:
                continue
            
            # 如果已存在该图片的结果，选择相似度更高的
            if unique_id in unique_results:
                existing_score = unique_results[unique_id].get("similarity_score", 0.0)
                current_score = result.get("similarity_score", 0.0)
                
                if current_score > existing_score:
                    # 保留匹配类型信息
                    result["all_match_types"] = unique_results[unique_id].get("all_match_types", []) + [unique_results[unique_id].get("match_type")]
                    result["all_match_types"].append(result.get("match_type"))
                    unique_results[unique_id] = result
                else:
                    # 更新匹配类型信息
                    unique_results[unique_id]["all_match_types"] = unique_results[unique_id].get("all_match_types", [unique_results[unique_id].get("match_type")])
                    unique_results[unique_id]["all_match_types"].append(result.get("match_type"))
            else:
                result["all_match_types"] = [result.get("match_type")]
                unique_results[unique_id] = result
        
        # 重新排序
        deduplicated_results = list(unique_results.values())
        deduplicated_results.sort(key=lambda x: x.get("similarity_score", 0.0), reverse=True)
        
        # 更新排名
        for i, result in enumerate(deduplicated_results):
            result["rank"] = i + 1
        
        logger.debug(f"去重处理: {len(results)} -> {len(deduplicated_results)}")
        return deduplicated_results
    
    def search_by_text(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        enable_filter: bool = True,
        enable_dedup: bool = True
    ) -> List[Dict[str, Any]]:
        """
        基于文本查询搜索图片
        
        Args:
            query_text: 查询文本
            top_k: 返回结果数量
            enable_filter: 是否启用相似度过滤
            enable_dedup: 是否启用去重
            
        Returns:
            搜索结果列表
        """
        try:
            if not query_text or not query_text.strip():
                logger.warning("查询文本为空")
                return []
            
            top_k = top_k or self.default_top_k
            logger.info(f"文本查询: '{query_text}', top_k: {top_k}")
            
            # 使用embedding处理器进行向量搜索
            raw_results = self.embedding_processor.search_by_text(query_text, top_k * 3)  # 获取更多结果用于后续处理
            
            if not raw_results:
                logger.info("未找到匹配结果")
                return []
            
            # 使用元数据增强结果
            enhanced_results = self._enhance_results_with_metadata(raw_results)
            
            # 应用过滤器
            if enable_filter:
                enhanced_results = self._filter_by_similarity(enhanced_results)
            
            # 应用去重
            if enable_dedup:
                enhanced_results = self._deduplicate_results(enhanced_results)
            
            # 限制最终结果数量
            final_results = enhanced_results[:top_k]
            
            logger.info(f"文本查询完成，返回 {len(final_results)} 个结果")
            return final_results
            
        except Exception as e:
            logger.error(f"文本查询失败: {e}")
            return []
    
    def search_by_image(
        self,
        image_input: Union[str, bytes],
        top_k: Optional[int] = None,
        enable_filter: bool = True,
        enable_dedup: bool = True
    ) -> List[Dict[str, Any]]:
        """
        基于图片查询搜索相似图片
        
        Args:
            image_input: 图片输入（文件路径或bytes数据）
            top_k: 返回结果数量
            enable_filter: 是否启用相似度过滤
            enable_dedup: 是否启用去重
            
        Returns:
            搜索结果列表
        """
        try:
            top_k = top_k or self.default_top_k
            logger.info(f"图片查询, top_k: {top_k}")
            
            # 处理图片输入
            if isinstance(image_input, str):
                # 文件路径
                if not Path(image_input).exists():
                    logger.error(f"图片文件不存在: {image_input}")
                    return []
                image_base64 = image_to_base64(image_input, max_size=(1024, 1024))
            elif isinstance(image_input, bytes):
                # 直接的图片bytes数据
                import base64
                image_base64 = base64.b64encode(image_input).decode('utf-8')
            else:
                logger.error("不支持的图片输入类型")
                return []
            
            # 使用embedding处理器进行向量搜索
            raw_results = self.embedding_processor.search_by_image(image_base64, top_k * 3)
            
            if not raw_results:
                logger.info("未找到匹配结果")
                return []
            
            # 使用元数据增强结果
            enhanced_results = self._enhance_results_with_metadata(raw_results)
            
            # 应用过滤器
            if enable_filter:
                enhanced_results = self._filter_by_similarity(enhanced_results)
            
            # 应用去重
            if enable_dedup:
                enhanced_results = self._deduplicate_results(enhanced_results)
            
            # 限制最终结果数量
            final_results = enhanced_results[:top_k]
            
            logger.info(f"图片查询完成，返回 {len(final_results)} 个结果")
            return final_results
            
        except Exception as e:
            logger.error(f"图片查询失败: {e}")
            return []
    
    def hybrid_search(
        self,
        query_text: Optional[str] = None,
        query_image: Optional[Union[str, bytes]] = None,
        top_k: Optional[int] = None,
        text_weight: float = 0.6,
        image_weight: float = 0.4
    ) -> List[Dict[str, Any]]:
        """
        混合搜索（文本+图片）
        
        Args:
            query_text: 查询文本
            query_image: 查询图片
            top_k: 返回结果数量
            text_weight: 文本权重
            image_weight: 图片权重
            
        Returns:
            搜索结果列表
        """
        try:
            if not query_text and not query_image:
                logger.warning("查询文本和图片都为空")
                return []
            
            top_k = top_k or self.default_top_k
            logger.info(f"混合查询, 文本权重: {text_weight}, 图片权重: {image_weight}")
            
            text_results = []
            image_results = []
            
            # 执行文本搜索
            if query_text:
                text_results = self.search_by_text(query_text, top_k * 2, enable_dedup=False)
            
            # 执行图片搜索
            if query_image:
                image_results = self.search_by_image(query_image, top_k * 2, enable_dedup=False)
            
            # 合并和重新评分
            combined_results = self._combine_search_results(
                text_results, image_results, text_weight, image_weight
            )
            
            # 去重和排序
            final_results = self._deduplicate_results(combined_results)
            final_results = final_results[:top_k]
            
            logger.info(f"混合查询完成，返回 {len(final_results)} 个结果")
            return final_results
            
        except Exception as e:
            logger.error(f"混合查询失败: {e}")
            return []
    
    def _combine_search_results(
        self,
        text_results: List[Dict[str, Any]],
        image_results: List[Dict[str, Any]],
        text_weight: float,
        image_weight: float
    ) -> List[Dict[str, Any]]:
        """
        合并文本和图片搜索结果
        
        Args:
            text_results: 文本搜索结果
            image_results: 图片搜索结果
            text_weight: 文本权重
            image_weight: 图片权重
            
        Returns:
            合并后的结果
        """
        combined_scores = {}
        
        # 处理文本结果
        for result in text_results:
            unique_id = result.get("unique_id", "")
            if unique_id:
                text_score = result.get("similarity_score", 0.0)
                combined_scores[unique_id] = {
                    "text_score": text_score,
                    "image_score": 0.0,
                    "result": result
                }
        
        # 处理图片结果
        for result in image_results:
            unique_id = result.get("unique_id", "")
            if unique_id:
                image_score = result.get("similarity_score", 0.0)
                if unique_id in combined_scores:
                    combined_scores[unique_id]["image_score"] = image_score
                else:
                    combined_scores[unique_id] = {
                        "text_score": 0.0,
                        "image_score": image_score,
                        "result": result
                    }
        
        # 计算综合分数
        combined_results = []
        for unique_id, scores in combined_scores.items():
            combined_score = (
                scores["text_score"] * text_weight +
                scores["image_score"] * image_weight
            )
            
            result = scores["result"].copy()
            result["combined_score"] = combined_score
            result["text_score"] = scores["text_score"]
            result["image_score"] = scores["image_score"]
            result["similarity_score"] = combined_score  # 使用综合分数作为主要相似度
            
            combined_results.append(result)
        
        # 按综合分数排序
        combined_results.sort(key=lambda x: x.get("combined_score", 0.0), reverse=True)
        
        # 更新排名
        for i, result in enumerate(combined_results):
            result["rank"] = i + 1
        
        return combined_results
    
    def get_similar_images(self, target_image_path: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        获取与目标图片相似的图片
        
        Args:
            target_image_path: 目标图片路径
            top_k: 返回数量
            
        Returns:
            相似图片列表
        """
        return self.search_by_image(target_image_path, top_k, enable_filter=False)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取检索系统统计信息
        
        Returns:
            统计信息字典
        """
        embedding_stats = self.embedding_processor.get_statistics()
        
        return {
            "metadata_count": len(self.metadata_index),
            "vector_count": embedding_stats.get("total_vectors", 0),
            "index_dimension": embedding_stats.get("embedding_dimension", 0),
            "similarity_threshold": self.similarity_threshold,
            "default_top_k": self.default_top_k,
            "metadata_path": self.metadata_path,
            "config": {
                "enable_rerank": self.enable_rerank,
                "vector_store_config": embedding_stats.get("config", {})
            }
        }
    
    def reload_metadata(self) -> bool:
        """
        重新加载元数据索引
        
        Returns:
            是否加载成功
        """
        try:
            old_count = len(self.metadata_index)
            self.metadata_index = self._load_metadata_index()
            new_count = len(self.metadata_index)
            
            logger.info(f"元数据重新加载完成: {old_count} -> {new_count}")
            return True
            
        except Exception as e:
            logger.error(f"重新加载元数据失败: {e}")
            return False