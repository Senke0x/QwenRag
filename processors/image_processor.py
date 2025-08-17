"""
封装 Qwen-VL 的图像识别和分析逻辑
"""
import json
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from openai import OpenAI
import requests

from schemas.data_models import ImageMetadata, ProcessingStatus
from config import QwenVLConfig, RetryConfig
from utils.retry_utils import retry_with_backoff, RetryableError, NonRetryableError
from utils.image_utils import (
    validate_image_file, image_to_base64, crop_face_from_image,
    extract_image_timestamp, generate_image_id, get_image_info
)

logger = logging.getLogger(__name__)


class QwenVLError(Exception):
    """Qwen VL API相关错误的基类"""
    pass


class QwenVLAuthError(NonRetryableError):
    """认证错误，不可重试"""
    pass


class QwenVLRateLimitError(RetryableError):
    """限流错误，可重试"""
    pass


class QwenVLServiceError(RetryableError):
    """服务错误，可重试"""
    pass


class ImageProcessor:
    """图片处理器，封装Qwen VL API调用"""
    
    def __init__(
        self,
        qwen_config: Optional[QwenVLConfig] = None,
        retry_config: Optional[RetryConfig] = None
    ):
        """
        初始化图片处理器
        
        Args:
            qwen_config: Qwen VL配置
            retry_config: 重试配置
        """
        from config import config as default_config
        
        self.qwen_config = qwen_config or default_config.qwen_vl
        self.retry_config = retry_config or default_config.retry
        self.image_config = default_config.image_processor
        
        # 验证API密钥
        if not self.qwen_config.api_key:
            raise ValueError("DASHSCOPE_API_KEY环境变量未设置或为空")
        
        # 初始化OpenAI客户端（兼容模式）
        self.client = OpenAI(
            api_key=self.qwen_config.api_key,
            base_url=self.qwen_config.base_url
        )
        
        logger.info(f"图片处理器初始化完成，模型: {self.qwen_config.model}")
    
    def _handle_api_error(self, error: Exception) -> Exception:
        """
        处理API错误，转换为对应的异常类型
        
        Args:
            error: 原始异常
            
        Returns:
            转换后的异常
        """
        if hasattr(error, 'response'):
            status_code = getattr(error.response, 'status_code', 0)
            
            if status_code == 401:
                return QwenVLAuthError(f"API认证失败: {error}")
            elif status_code == 429:
                return QwenVLRateLimitError(f"API限流: {error}")
            elif status_code >= 500:
                return QwenVLServiceError(f"服务错误: {error}")
            elif 400 <= status_code < 500:
                return NonRetryableError(f"客户端错误: {error}")
        
        # 网络相关错误
        error_str = str(error).lower()
        if any(keyword in error_str for keyword in ['timeout', 'connection', 'network']):
            return RetryableError(f"网络错误: {error}")
        
        return QwenVLError(f"未知错误: {error}")
    
    @retry_with_backoff()
    def _call_qwen_vl_api(self, image_base64: str, prompt: str) -> Dict[str, Any]:
        """
        调用Qwen VL API进行图片分析
        
        Args:
            image_base64: 图片的base64编码
            prompt: 分析提示词
            
        Returns:
            API响应结果
        """
        try:
            messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "你是一个专业的图像分析助手。请仔细分析图片内容，并以JSON格式返回结果。"
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
            
            response = self.client.chat.completions.create(
                model=self.qwen_config.model,
                messages=messages,
                max_tokens=self.qwen_config.max_tokens,
                temperature=self.qwen_config.temperature,
                timeout=self.qwen_config.timeout
            )
            
            content = response.choices[0].message.content
            logger.debug(f"Qwen VL API响应: {content}")
            
            return content
            
        except Exception as e:
            logger.error(f"Qwen VL API调用失败: {e}")
            raise self._handle_api_error(e)
    
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
        
        # 构造分析提示词
        prompt = """
请仔细分析这张图片，并以JSON格式返回以下信息：
{
    "is_snap": boolean,  // 是否是手机截图
    "is_landscape": boolean,  // 是否是风景照
    "description": "string",  // 详细的图片描述，用于语义检索
    "has_person": boolean,  // 是否有人物
    "face_rects": [[x,y,w,h], ...]  // 人脸位置框，格式为[x,y,width,height]的数组
}

注意：
1. is_snap: 判断是否为手机应用界面截图、网页截图等
2. is_landscape: 判断是否为自然风景、山水、城市景观等
3. description: 提供详细的中文描述，包括主要内容、颜色、场景等
4. has_person: 判断图片中是否包含人物（包括部分身体）
5. face_rects: 如果有人脸，提供人脸的边界框坐标

请只返回JSON格式的结果，不要包含其他文字。
"""
        
        # 调用API
        content = self._call_qwen_vl_api(image_base64, prompt)
        
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