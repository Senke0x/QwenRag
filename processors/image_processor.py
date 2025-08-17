"""
封装 Qwen-VL 的图像识别和分析逻辑
"""
import json
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

from schemas.data_models import ImageMetadata, ProcessingStatus
from config import QwenVLConfig, RetryConfig
from clients.qwen_client import QwenClient
from clients.prompt_manager import PromptManager, PromptType
from utils.image_utils import (
    validate_image_file, image_to_base64, crop_face_from_image,
    extract_image_timestamp, generate_image_id, get_image_info
)

logger = logging.getLogger(__name__)




class ImageProcessor:
    """图片处理器，封装Qwen VL API调用"""
    
    def __init__(
        self,
        qwen_client: Optional[QwenClient] = None,
        prompt_manager: Optional[PromptManager] = None
    ):
        """
        初始化图片处理器
        
        Args:
            qwen_client: Qwen 客户端实例
            prompt_manager: 提示词管理器实例
        """
        from config import config as default_config
        
        self.qwen_client = qwen_client or QwenClient()
        self.prompt_manager = prompt_manager or PromptManager()
        self.image_config = default_config.image_processor
        
        logger.info("图片处理器初始化完成")
    
    
    def _call_qwen_vl_api(self, image_base64: str, prompt_type: PromptType = PromptType.IMAGE_ANALYSIS) -> str:
        """
        调用Qwen VL API进行图片分析
        
        Args:
            image_base64: 图片的base64编码
            prompt_type: 提示词类型
            
        Returns:
            API响应结果
        """
        system_prompt = self.prompt_manager.get_system_prompt(prompt_type)
        user_prompt = self.prompt_manager.get_user_prompt(prompt_type)
        
        return self.qwen_client.chat_with_image(
            image_base64=image_base64,
            user_prompt=user_prompt,
            system_prompt=system_prompt
        )
    
    def _parse_analysis_result(self, content: str) -> Dict[str, Any]:
        """
        解析API返回的分析结果
        
        Args:
            content: API返回的内容
            
        Returns:
            解析后的结果字典
        """
        try:
            # 尝试直接解析JSON
            if content.strip().startswith('{'):
                return json.loads(content)
            
            # 如果不是直接的JSON，尝试提取JSON部分
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
            # 如果无法解析为JSON，返回默认结构
            logger.warning(f"无法解析API响应为JSON，使用默认值: {content}")
            return {
                "is_snap": False,
                "is_landscape": False,
                "description": content[:200],  # 截取前200字符作为描述
                "has_person": False,
                "face_rects": []
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}, 内容: {content}")
            # 返回默认结构
            return {
                "is_snap": False,
                "is_landscape": False,
                "description": "图片分析失败",
                "has_person": False,
                "face_rects": []
            }
    
    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """
        分析单张图片
        
        Args:
            image_path: 图片路径
            
        Returns:
            分析结果字典
        """
        # 验证图片文件
        is_valid, error_msg = validate_image_file(image_path, self.image_config)
        if not is_valid:
            raise ValueError(f"图片验证失败: {error_msg}")
        
        # 转换为base64
        image_base64 = image_to_base64(image_path, max_size=(1024, 1024))
        
        # 调用API
        content = self._call_qwen_vl_api(image_base64, PromptType.IMAGE_ANALYSIS)
        
        # 解析结果
        return self._parse_analysis_result(content)
    
    def process_image(self, image_path: str) -> ImageMetadata:
        """
        处理单张图片，生成完整的元数据
        
        Args:
            image_path: 图片路径
            
        Returns:
            图片元数据对象
        """
        metadata = ImageMetadata(path=image_path)
        
        try:
            # 获取基本图片信息
            img_info = get_image_info(image_path)
            metadata.unique_id = img_info['unique_id']
            metadata.timestamp = img_info['timestamp']
            
            # 分析图片内容
            analysis_result = self.analyze_image(image_path)
            
            # 更新元数据
            metadata.is_snap = analysis_result.get('is_snap', False)
            metadata.is_landscape = analysis_result.get('is_landscape', False)
            metadata.description = analysis_result.get('description', '')
            metadata.has_person = analysis_result.get('has_person', False)
            metadata.face_rects = analysis_result.get('face_rects', [])
            
            # 设置成功状态
            metadata.processing_status = ProcessingStatus.SUCCESS
            metadata.last_processed = datetime.now()
            
            logger.info(f"图片处理成功: {image_path}")
            
        except Exception as e:
            # 设置失败状态
            metadata.processing_status = ProcessingStatus.FAILED
            metadata.error_message = str(e)
            metadata.last_processed = datetime.now()
            
            logger.error(f"图片处理失败: {image_path}, 错误: {e}")
        
        return metadata
    
    def process_images_batch(self, image_paths: List[str]) -> List[ImageMetadata]:
        """
        批量处理图片
        
        Args:
            image_paths: 图片路径列表
            
        Returns:
            图片元数据列表
        """
        results = []
        
        for i, image_path in enumerate(image_paths):
            logger.info(f"处理图片 {i+1}/{len(image_paths)}: {image_path}")
            
            try:
                metadata = self.process_image(image_path)
                results.append(metadata)
                
            except Exception as e:
                logger.error(f"批量处理失败: {image_path}, 错误: {e}")
                # 创建失败的元数据记录
                metadata = ImageMetadata(
                    path=image_path,
                    processing_status=ProcessingStatus.FAILED,
                    error_message=str(e),
                    last_processed=datetime.now()
                )
                results.append(metadata)
        
        return results
    
    def extract_face_embeddings(self, metadata: ImageMetadata) -> List[str]:
        """
        提取人脸区域的base64编码
        
        Args:
            metadata: 图片元数据
            
        Returns:
            人脸区域的base64编码列表
        """
        face_base64_list = []
        
        if not metadata.has_person or not metadata.face_rects:
            return face_base64_list
        
        try:
            for face_rect in metadata.face_rects:
                face_base64 = crop_face_from_image(metadata.path, face_rect)
                face_base64_list.append(face_base64)
                
            logger.info(f"提取到 {len(face_base64_list)} 个人脸区域: {metadata.path}")
            
        except Exception as e:
            logger.error(f"人脸提取失败: {metadata.path}, 错误: {e}")
        
        return face_base64_list