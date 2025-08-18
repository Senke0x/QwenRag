"""
Embedding处理器 - 负责向量化和存储管理
"""
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np

from clients.qwen_client import QwenClient
from vector_store.faiss_store import FaissStore
from utils.logger import logger

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
        embedding_dimension: int = 1024,  # dashscope embedding模型的维度
        index_save_path: str = "data/faiss_index"
    ):
        """
        初始化Embedding处理器
        
        Args:
            qwen_client: Qwen客户端实例
            vector_store: FAISS存储实例  
            embedding_dimension: 向量维度
            index_save_path: 索引保存路径
        """
        self.qwen_client = qwen_client or QwenClient()
        self.vector_store = vector_store or FaissStore(dimension=embedding_dimension)
        self.embedding_dimension = embedding_dimension
        self.index_save_path = index_save_path
        
        # 确保保存目录存在
        Path(index_save_path).parent.mkdir(parents=True, exist_ok=True)
        
        # 加载已有索引（如果存在）
        self._load_existing_index()
        
        logger.info(f"EmbeddingProcessor初始化完成，维度: {embedding_dimension}")
    
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
            text_vector = np.array(response['embedding'], dtype=np.float32)
            
            # 验证并调整维度
            if text_vector.shape[0] != self.embedding_dimension:
                logger.warning(f"向量维度不匹配，自动调整: 期望{self.embedding_dimension}, 实际{text_vector.shape[0]}")
                self.embedding_dimension = text_vector.shape[0]
            
            # 存储向量
            success = self.vector_store.add_vectors(
                vectors=text_vector.reshape(1, -1),
                ids=[text_id]
            )
            
            if success:
                logger.info(f"成功处理文本数据: {text_id}")
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
            image_vector = np.array(response['embedding'], dtype=np.float32)
            
            # 验证并调整维度
            if image_vector.shape[0] != self.embedding_dimension:
                logger.warning(f"向量维度不匹配，自动调整: 期望{self.embedding_dimension}, 实际{image_vector.shape[0]}")
                self.embedding_dimension = image_vector.shape[0]
            
            # 存储向量
            success = self.vector_store.add_vectors(
                vectors=image_vector.reshape(1, -1),
                ids=[image_id]
            )
            
            if success:
                logger.info(f"成功处理base64图片: {image_id}")
                self.save_index()
                return True
            else:
                logger.error(f"base64图片处理失败: {image_id}")
                return False
                
        except Exception as e:
            logger.error(f"处理base64图片失败: {image_id}, 错误: {e}")
            return False
    
    def process_image_with_faces(self, image_base64: str, face_rects: List[Dict[str, int]], image_id: str) -> bool:
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
            whole_image_vector = np.array(whole_image_result['embedding'], dtype=np.float32)
            vectors_to_store.append(whole_image_vector)
            ids_to_store.append(f"{image_id}_full")
            
            # 处理每个人脸区域
            for i, face_rect in enumerate(face_rects):
                try:
                    face_result = self.qwen_client.get_face_embedding(image_base64, face_rect)
                    face_vector = np.array(face_result['embedding'], dtype=np.float32)
                    vectors_to_store.append(face_vector)
                    ids_to_store.append(f"{image_id}_face_{i}")
                    logger.debug(f"成功处理人脸{i}: {image_id}")
                except Exception as e:
                    logger.warning(f"处理人脸{i}失败: {e}")
                    # 继续处理其他人脸
                    continue
            
            # 验证并调整维度
            if vectors_to_store and vectors_to_store[0].shape[0] != self.embedding_dimension:
                logger.warning(f"向量维度不匹配，自动调整: 期望{self.embedding_dimension}, 实际{vectors_to_store[0].shape[0]}")
                self.embedding_dimension = vectors_to_store[0].shape[0]
            
            # 批量存储向量
            if vectors_to_store:
                vectors_array = np.vstack(vectors_to_store)
                success = self.vector_store.add_vectors(vectors_array, ids_to_store)
                
                if success:
                    logger.info(f"成功存储 {len(vectors_to_store)} 个向量: {image_id}")
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
    
    def process_batch_texts(self, texts: List[str], text_ids: List[str]) -> Dict[str, Any]:
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
        
        results = {
            "total": len(texts),
            "success": 0,
            "failed": 0,
            "failed_items": []
        }
        
        logger.info(f"开始批量处理 {len(texts)} 个文本")
        
        for i, (text, text_id) in enumerate(zip(texts, text_ids)):
            logger.info(f"处理进度: {i+1}/{len(texts)} - {text_id}")
            
            if self.process_text(text, text_id):
                results["success"] += 1
            else:
                results["failed"] += 1
                results["failed_items"].append(text_id)
            
            # 每处理10个就保存一次索引
            if (i + 1) % 10 == 0:
                logger.info(f"处理了{i+1}个项目，保存索引...")
                self.save_index()
        
        # 最终保存索引
        if results["success"] > 0:
            logger.info("批量处理完成，保存最终索引...")
            self.save_index()
            
        logger.info(f"批量处理完成: 成功{results['success']}, 失败{results['failed']}")
        return results
    
    def process_batch_images(self, images_base64: List[str], image_ids: List[str]) -> Dict[str, Any]:
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
            "failed_items": []
        }
        
        logger.info(f"开始批量处理 {len(images_base64)} 个图片")
        
        for i, (image_base64, image_id) in enumerate(zip(images_base64, image_ids)):
            logger.info(f"处理进度: {i+1}/{len(images_base64)} - {image_id}")
            
            if self.process_image_base64(image_base64, image_id):
                results["success"] += 1
            else:
                results["failed"] += 1
                results["failed_items"].append(image_id)
            
            # 每处理10个就保存一次索引
            if (i + 1) % 10 == 0:
                logger.info(f"处理了{i+1}个项目，保存索引...")
                self.save_index()
        
        # 最终保存索引
        if results["success"] > 0:
            logger.info("批量处理完成，保存最终索引...")
            self.save_index()
            
        logger.info(f"批量处理完成: 成功{results['success']}, 失败{results['failed']}")
        return results
    
    def search_by_text(self, query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        基于文本查询搜索相似向量
        
        Args:
            query_text: 查询文本
            top_k: 返回结果数量
            
        Returns:
            搜索结果列表
        """
        try:
            logger.info(f"文本搜索: {query_text}, top_k: {top_k}")
            
            # 生成查询向量
            response = self.qwen_client.get_text_embedding(query_text)
            query_vector = np.array(response['embedding'], dtype=np.float32)
            
            # 搜索相似向量
            distances, indices, ids = self.vector_store.search(
                query_vectors=query_vector.reshape(1, -1),
                k=top_k
            )
            
            # 整理搜索结果
            results = []
            for i, (distance, _, vector_id) in enumerate(zip(distances, indices, ids)):
                if vector_id:  # 确保ID有效
                    similarity_score = float(1.0 / (1.0 + distance)) if distance > 0 else 1.0
                    
                    results.append({
                        "vector_id": vector_id,
                        "similarity_score": similarity_score,
                        "distance": float(distance),
                        "rank": i + 1
                    })
            
            logger.info(f"文本搜索完成，返回 {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"文本搜索失败: {e}")
            return []
    
    def search_by_image(self, image_base64: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        基于图片查询搜索相似向量
        
        Args:
            image_base64: 查询图片base64
            top_k: 返回结果数量
            
        Returns:
            搜索结果列表
        """
        try:
            logger.info(f"图片搜索, top_k: {top_k}")
            
            # 生成查询向量
            response = self.qwen_client.get_image_embedding(image_base64)
            query_vector = np.array(response['embedding'], dtype=np.float32)
            
            # 搜索相似向量
            distances, indices, ids = self.vector_store.search(
                query_vectors=query_vector.reshape(1, -1),
                k=top_k
            )
            
            # 整理搜索结果
            results = []
            for i, (distance, _, vector_id) in enumerate(zip(distances, indices, ids)):
                if vector_id:
                    similarity_score = float(1.0 / (1.0 + distance)) if distance > 0 else 1.0
                    
                    results.append({
                        "vector_id": vector_id,
                        "similarity_score": similarity_score,
                        "distance": float(distance),
                        "rank": i + 1
                    })
            
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
            "id_mapping_size": vector_stats.get("id_mapping_size", 0)
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