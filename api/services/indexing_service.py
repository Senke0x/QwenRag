"""
索引管理业务服务
"""
import time
import os
from typing import List, Dict, Any, Optional
from pathlib import Path

from pipelines.indexing_pipeline import IndexingPipeline
from processors.face_processor import FaceProcessor
from utils.logger import setup_logger

logger = setup_logger(__name__)

class IndexingService:
    """索引管理业务服务"""
    
    def __init__(
        self, 
        indexing_pipeline: IndexingPipeline,
        face_processor: FaceProcessor
    ):
        self.indexing_pipeline = indexing_pipeline
        self.face_processor = face_processor
        self.background_tasks: Dict[str, Dict[str, Any]] = {}  # 后台任务状态存储
        
        logger.info("索引服务初始化完成")
    
    async def get_index_status(self) -> Dict[str, Any]:
        """
        获取索引状态信息
        
        Returns:
            索引状态字典
        """
        try:
            logger.info("获取索引状态")
            
            # 获取图像索引统计
            image_stats = self.indexing_pipeline.get_statistics()
            
            # 获取人脸索引统计
            face_stats = self.face_processor.get_face_statistics()
            
            # 计算索引文件大小
            index_size_mb = 0
            try:
                # 检查各种索引文件的大小
                index_files = [
                    "data/faiss_index",
                    "data/face_faiss_index", 
                    "data/face_metadata.json",
                    "index_metadata.json"
                ]
                
                for file_path in index_files:
                    if os.path.exists(file_path):
                        index_size_mb += os.path.getsize(file_path) / (1024 * 1024)
            except Exception as e:
                logger.warning(f"计算索引文件大小失败: {e}")
            
            # 构建状态信息
            status_info = {
                "total_images": image_stats.get("total_processed", 0),
                "indexed_images": image_stats.get("success_count", 0),
                "total_faces": face_stats.get("store_stats", {}).get("total_faces", 0),
                "index_size_mb": round(index_size_mb, 2),
                "last_updated": image_stats.get("end_time"),
                "index_health": self._assess_index_health(image_stats, face_stats),
                "image_statistics": image_stats,
                "face_statistics": face_stats
            }
            
            logger.info(f"索引状态获取完成: {status_info['indexed_images']}张图片已索引")
            return status_info
            
        except Exception as e:
            logger.error(f"获取索引状态失败: {e}")
            raise
    
    def _assess_index_health(
        self, 
        image_stats: Dict[str, Any], 
        face_stats: Dict[str, Any]
    ) -> str:
        """评估索引健康状态"""
        try:
            success_count = image_stats.get("success_count", 0)
            total_processed = image_stats.get("total_processed", 0)
            failed_count = image_stats.get("failed_count", 0)
            
            if total_processed == 0:
                return "empty"
            
            success_rate = success_count / total_processed if total_processed > 0 else 0
            
            if success_rate >= 0.95:
                return "healthy"
            elif success_rate >= 0.8:
                return "warning" 
            else:
                return "unhealthy"
                
        except Exception:
            return "unknown"
    
    async def build_index(
        self,
        image_directory: str,
        batch_size: int = 10,
        max_workers: int = 4,
        force_rebuild: bool = False,
        process_faces: bool = True
    ) -> Dict[str, Any]:
        """
        构建图片索引
        
        Args:
            image_directory: 图片目录路径
            batch_size: 批处理大小
            max_workers: 最大并发数
            force_rebuild: 是否强制重建
            process_faces: 是否处理人脸
            
        Returns:
            构建结果字典
        """
        try:
            logger.info(f"开始构建索引: {image_directory}")
            
            # 重置统计信息
            if force_rebuild:
                self.indexing_pipeline.reset_statistics()
            
            # 扫描目录下的图片文件
            from utils.image_utils import get_supported_image_extensions
            
            supported_extensions = [f".{ext}" for ext in get_supported_image_extensions()]
            image_files = []
            
            for ext in supported_extensions:
                image_files.extend(Path(image_directory).rglob(f"*{ext}"))
            
            image_paths = [str(path) for path in image_files]
            logger.info(f"找到{len(image_paths)}个图片文件")
            
            if not image_paths:
                return {
                    "total_images": 0,
                    "processed_images": 0,
                    "failed_images": 0,
                    "message": "没有找到支持的图片文件"
                }
            
            # 执行批量处理
            processing_result = self.indexing_pipeline.process_image_batch(
                image_paths=image_paths,
                parallel=max_workers > 1
            )
            
            # 处理人脸embeddings（如果启用）
            face_processing_result = {}
            if process_faces:
                try:
                    logger.info("开始处理人脸embeddings")
                    face_processing_result = self.indexing_pipeline.process_pending_face_embeddings()
                    logger.info(f"人脸processing完成: {face_processing_result}")
                except Exception as e:
                    logger.error(f"人脸embedding处理失败: {e}")
                    face_processing_result = {"error": str(e)}
            
            # 保存元数据
            self.indexing_pipeline.save_metadata()
            
            build_result = {
                "total_images": len(image_paths),
                "processed_images": processing_result.get("successful_images", 0),
                "failed_images": processing_result.get("failed_images", 0),
                "face_processing_results": face_processing_result
            }
            
            logger.info(f"索引构建完成: {build_result}")
            return build_result
            
        except Exception as e:
            logger.error(f"构建索引失败: {e}")
            raise
    
    async def build_index_background(
        self,
        task_id: str,
        image_directory: str,
        batch_size: int,
        max_workers: int,
        force_rebuild: bool,
        process_faces: bool
    ):
        """
        后台构建索引任务
        
        Args:
            task_id: 任务ID
            image_directory: 图片目录
            batch_size: 批处理大小
            max_workers: 最大并发数
            force_rebuild: 是否强制重建
            process_faces: 是否处理人脸
        """
        try:
            # 初始化任务状态
            self.background_tasks[task_id] = {
                "status": "running",
                "start_time": time.time(),
                "progress": 0,
                "message": "正在构建索引..."
            }
            
            logger.info(f"后台索引构建任务开始: {task_id}")
            
            # 执行索引构建
            build_result = await self.build_index(
                image_directory=image_directory,
                batch_size=batch_size,
                max_workers=max_workers,
                force_rebuild=force_rebuild,
                process_faces=process_faces
            )
            
            # 更新任务状态为完成
            self.background_tasks[task_id].update({
                "status": "completed",
                "end_time": time.time(),
                "progress": 100,
                "message": "索引构建完成",
                "result": build_result
            })
            
            logger.info(f"后台索引构建任务完成: {task_id}")
            
        except Exception as e:
            # 更新任务状态为失败
            self.background_tasks[task_id].update({
                "status": "failed",
                "end_time": time.time(),
                "message": f"索引构建失败: {str(e)}",
                "error": str(e)
            })
            
            logger.error(f"后台索引构建任务失败: {task_id}, 错误: {e}")
    
    async def add_images(
        self,
        image_paths: List[str],
        process_faces: bool = True
    ) -> Dict[str, Any]:
        """
        添加图片到现有索引
        
        Args:
            image_paths: 图片路径列表
            process_faces: 是否处理人脸
            
        Returns:
            添加结果字典
        """
        try:
            logger.info(f"添加图片到索引: {len(image_paths)}张图片")
            
            # 过滤有效的图片路径
            valid_paths = [path for path in image_paths if os.path.exists(path)]
            invalid_paths = [path for path in image_paths if path not in valid_paths]
            
            if invalid_paths:
                logger.warning(f"发现无效路径: {len(invalid_paths)}个")
            
            if not valid_paths:
                return {
                    "added_images": 0,
                    "failed_images": len(image_paths),
                    "image_details": [{"path": path, "status": "not_found"} for path in invalid_paths]
                }
            
            # 执行批量处理
            processing_result = self.indexing_pipeline.process_image_batch(
                image_paths=valid_paths,
                parallel=True
            )
            
            # 处理人脸embeddings（如果启用）
            if process_faces:
                try:
                    face_processing_result = self.indexing_pipeline.process_pending_face_embeddings()
                    logger.info(f"人脸处理结果: {face_processing_result}")
                except Exception as e:
                    logger.error(f"人脸处理失败: {e}")
            
            # 保存元数据
            self.indexing_pipeline.save_metadata()
            
            add_result = {
                "added_images": processing_result.get("successful_images", 0),
                "failed_images": processing_result.get("failed_images", 0) + len(invalid_paths),
                "image_details": []  # 可以添加详细的处理信息
            }
            
            logger.info(f"图片添加完成: {add_result}")
            return add_result
            
        except Exception as e:
            logger.error(f"添加图片失败: {e}")
            raise
    
    async def clear_index(self) -> Dict[str, Any]:
        """
        清空所有索引数据
        
        Returns:
            清理结果字典
        """
        try:
            logger.info("开始清空索引")
            
            clear_result = {
                "cleared_items": [],
                "errors": []
            }
            
            # 清空图像处理管道的数据
            try:
                self.indexing_pipeline.reset_statistics()
                clear_result["cleared_items"].append("image_pipeline_stats")
            except Exception as e:
                clear_result["errors"].append(f"清空图像管道统计失败: {e}")
            
            # 清空人脸处理器的数据
            try:
                # 重建空的人脸索引
                self.face_processor.face_store.face_metadata.clear()
                self.face_processor.face_store._init_index()
                self.face_processor.face_store.save_index()
                self.face_processor.face_store._save_face_metadata()
                clear_result["cleared_items"].append("face_processor_data")
            except Exception as e:
                clear_result["errors"].append(f"清空人脸数据失败: {e}")
            
            # 删除索引文件
            index_files = [
                "data/faiss_index",
                "data/face_faiss_index",
                "data/face_metadata.json", 
                "index_metadata.json"
            ]
            
            for file_path in index_files:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        clear_result["cleared_items"].append(file_path)
                except Exception as e:
                    clear_result["errors"].append(f"删除文件{file_path}失败: {e}")
            
            logger.info(f"索引清空完成: {clear_result}")
            return clear_result
            
        except Exception as e:
            logger.error(f"清空索引失败: {e}")
            raise
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        获取后台任务状态
        
        Args:
            task_id: 任务ID
            
        Returns:
            任务状态字典
        """
        try:
            if task_id in self.background_tasks:
                task_info = self.background_tasks[task_id].copy()
                
                # 计算运行时间
                if "start_time" in task_info:
                    if task_info["status"] == "running":
                        task_info["running_time"] = time.time() - task_info["start_time"]
                    elif "end_time" in task_info:
                        task_info["total_time"] = task_info["end_time"] - task_info["start_time"]
                
                return task_info
            else:
                return None
                
        except Exception as e:
            logger.error(f"获取任务状态失败: {e}")
            return None
    
    async def optimize_index(self) -> Dict[str, Any]:
        """
        优化索引性能
        
        Returns:
            优化结果字典
        """
        try:
            logger.info("开始索引优化")
            
            optimization_result = {
                "optimized_components": [],
                "improvements": {}
            }
            
            # 优化人脸索引
            try:
                logger.info("重建人脸索引以优化性能")
                self.face_processor.face_store.rebuild_index()
                optimization_result["optimized_components"].append("face_index")
            except Exception as e:
                logger.error(f"优化人脸索引失败: {e}")
            
            # TODO: 添加其他优化操作
            # 比如：压缩向量、清理无用数据、重新组织索引结构等
            
            logger.info(f"索引优化完成: {optimization_result}")
            return optimization_result
            
        except Exception as e:
            logger.error(f"索引优化失败: {e}")
            raise