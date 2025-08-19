"""
Qwen API 统一客户端
"""
import logging
import json
import sys
import base64
import io
from typing import Dict, Any, Optional
from openai import OpenAI
from PIL import Image
import dashscope
from http import HTTPStatus

from config import QwenVLConfig, RetryConfig
from utils.retry_utils import retry_with_backoff, RetryableError, NonRetryableError

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


class QwenClient:
    """Qwen API 统一客户端"""
    
    def __init__(
        self,
        qwen_config: Optional[QwenVLConfig] = None,
        retry_config: Optional[RetryConfig] = None,
        enable_logging: bool = True
    ):
        """
        初始化 Qwen 客户端
        
        Args:
            qwen_config: Qwen VL配置
            retry_config: 重试配置
            enable_logging: 是否启用请求/响应日志
        """
        # 直接从根级config.py导入，避免循环导入问题
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        
        # 直接从根级config.py导入
        import importlib.util
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.py')
        spec = importlib.util.spec_from_file_location("root_config", config_path)
        root_config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(root_config)
        
        # 尝试获取默认配置
        try:
            if hasattr(root_config, 'config') and root_config.config and hasattr(root_config.config, 'qwen_vl'):
                default_qwen_config = root_config.config.qwen_vl
            else:
                default_qwen_config = root_config.QwenVLConfig()
        except:
            default_qwen_config = root_config.QwenVLConfig()
        
        self.qwen_config = qwen_config or default_qwen_config
        
        # 处理retry_config，使用同样的方式
        try:
            if hasattr(root_config, 'config') and root_config.config and hasattr(root_config.config, 'retry'):
                default_retry_config = root_config.config.retry
            else:
                default_retry_config = root_config.RetryConfig()
        except:
            default_retry_config = root_config.RetryConfig()
        
        self.retry_config = retry_config or default_retry_config
        self.enable_logging = enable_logging
        
        # 验证API密钥
        if not self.qwen_config.api_key:
            raise ValueError("DASHSCOPE_API_KEY环境变量未设置或为空")
        
        # 初始化OpenAI客户端（兼容模式）
        self.client = OpenAI(
            api_key=self.qwen_config.api_key,
            base_url=self.qwen_config.base_url
        )
        
        # 设置dashscope API密钥
        dashscope.api_key = self.qwen_config.api_key
        
        logger.info(f"Qwen客户端初始化完成，模型: {self.qwen_config.model}")
    
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
    
    def _mask_image_data(self, data: Any) -> Any:
        """
        将请求数据中的图片base64替换为mock数据以简化日志输出
        
        Args:
            data: 原始数据
            
        Returns:
            处理后的数据
        """
        if isinstance(data, dict):
            result = {}
            for key, value in data.items():
                if key == "image_url" and isinstance(value, dict) and "url" in value:
                    # 替换图片base64数据
                    url = value["url"]
                    if url.startswith("data:image/"):
                        # 提取图片格式和计算数据长度
                        parts = url.split(",", 1)
                        if len(parts) == 2:
                            header, base64_data = parts
                            data_size = len(base64_data)
                            result[key] = {
                                **value,
                                "url": f"{header},<BASE64_IMAGE_DATA_{data_size}_BYTES>"
                            }
                        else:
                            result[key] = {"url": "<INVALID_IMAGE_URL>"}
                    else:
                        result[key] = value
                else:
                    result[key] = self._mask_image_data(value)
            return result
        elif isinstance(data, list):
            return [self._mask_image_data(item) for item in data]
        else:
            return data
    
    def _log_request_response(self, request_data: Dict[str, Any], response_content: str, method: str):
        """
        记录请求和响应日志
        
        Args:
            request_data: 请求数据
            response_content: 响应内容
            method: 方法名称
        """
        if not self.enable_logging:
            return
            
        # 处理请求数据，替换图片base64
        masked_request = self._mask_image_data(request_data)
        
        # 构建日志输出
        log_message = f"""
🚀 === {method.upper()} API调用 ===
📤 REQUEST:
{json.dumps(masked_request, indent=2, ensure_ascii=False)}

📥 RESPONSE:
{response_content}
=== API调用结束 ===
"""
        
        # 使用INFO级别输出
        logger.info(log_message)
        
        # 如果是在测试环境，直接打印到控制台（避免重复）
        import os
        if os.getenv('PYTEST_CURRENT_TEST') or 'pytest' in sys.modules:
            print(log_message)
        elif logger.level <= logging.INFO:
            # 如果logger配置了INFO级别，避免重复打印
            pass
        else:
            # 否则直接打印
            print(log_message)
    
    @retry_with_backoff()
    def chat_with_image(
        self, 
        image_base64: str, 
        user_prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        使用图片进行聊天
        
        Args:
            image_base64: 图片的base64编码
            user_prompt: 用户提示词
            system_prompt: 系统提示词
            **kwargs: 其他参数（如temperature, max_tokens等）
            
        Returns:
            API响应内容
        """
        try:
            messages = []
            
            # 添加系统提示词
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}]
                })
            
            # 添加用户消息（图片+文本）
            messages.append({
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
                        "text": user_prompt
                    }
                ]
            })
            
            # 合并参数
            call_params = {
                "model": self.qwen_config.model,
                "messages": messages,
                "max_tokens": self.qwen_config.max_tokens,
                "temperature": self.qwen_config.temperature,
                "timeout": self.qwen_config.timeout
            }
            call_params.update(kwargs)
            
            response = self.client.chat.completions.create(**call_params)
            
            content = response.choices[0].message.content
            
            # 记录请求和响应日志
            self._log_request_response(call_params, content, "chat_with_image")
            
            return content
            
        except Exception as e:
            logger.error(f"Qwen VL API调用失败: {e}")
            raise self._handle_api_error(e)
    
    @retry_with_backoff()
    def chat_with_text(
        self, 
        user_prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        纯文本聊天
        
        Args:
            user_prompt: 用户提示词
            system_prompt: 系统提示词
            **kwargs: 其他参数
            
        Returns:
            API响应内容
        """
        try:
            messages = []
            
            # 添加系统提示词
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            # 添加用户消息
            messages.append({
                "role": "user",
                "content": user_prompt
            })
            
            # 合并参数
            call_params = {
                "model": self.qwen_config.model,
                "messages": messages,
                "max_tokens": self.qwen_config.max_tokens,
                "temperature": self.qwen_config.temperature,
                "timeout": self.qwen_config.timeout
            }
            call_params.update(kwargs)
            
            response = self.client.chat.completions.create(**call_params)
            
            content = response.choices[0].message.content
            
            # 记录请求和响应日志
            self._log_request_response(call_params, content, "chat_with_text")
            
            return content
            
        except Exception as e:
            logger.error(f"Qwen API调用失败: {e}")
            raise self._handle_api_error(e)
    
    @retry_with_backoff()
    def get_text_embedding(self, text: str, model: str = "text-embedding-v4") -> Dict[str, Any]:
        """
        获取文本的embedding向量（使用dashscope SDK）
        
        Args:
            text: 要转换的文本
            model: embedding模型名称，默认使用text-embedding-v4
            
        Returns:
            包含embedding向量的响应字典
        """
        try:
            # 使用dashscope SDK调用text embedding
            resp = dashscope.TextEmbedding.call(
                model=model,
                input=text
            )
            
            if resp.status_code == HTTPStatus.OK:
                # 提取embedding向量（假设是单个文本）
                embedding_vector = resp.output['embeddings'][0]['embedding']
                
                embedding_data = {
                    "embedding": embedding_vector,
                    "model": model,
                    "status_code": resp.status_code,
                    "request_id": getattr(resp, "request_id", ""),
                    "usage": resp.usage if hasattr(resp, 'usage') else None
                }
                
                # 记录请求和响应日志
                request_data = {
                    "model": model,
                    "input": text[:100] + "..." if len(text) > 100 else text,
                    "sdk": "dashscope"
                }
                
                response_content = f"embedding_dimension: {len(embedding_vector)}, model: {model}, status: {resp.status_code}"
                self._log_request_response(request_data, response_content, "get_text_embedding")
                
                return embedding_data
            else:
                error_msg = f"Text embedding failed: status_code={resp.status_code}, code={getattr(resp, 'code', '')}, message={getattr(resp, 'message', '')}"
                logger.error(error_msg)
                raise QwenVLError(error_msg)
            
        except Exception as e:
            logger.error(f"获取文本embedding失败: {e}")
            raise self._handle_api_error(e)
    
    @retry_with_backoff()
    def get_image_embedding(self, image_base64: str, model: str = "multimodal-embedding-v1") -> Dict[str, Any]:
        """
        获取图片的embedding向量（使用dashscope multimodal embedding）
        
        Args:
            image_base64: 图片的base64编码
            model: embedding模型名称，默认使用multimodal-embedding-v1
            
        Returns:
            包含embedding向量的响应字典
        """
        try:
            # 构建图片数据格式（按照您提供的示例格式）
            image_data = f"data:image/jpeg;base64,{image_base64}"
            input_data = [{'image': image_data}]
            
            # 使用dashscope SDK调用multimodal embedding
            resp = dashscope.MultiModalEmbedding.call(
                model=model,
                input=input_data
            )
            
            if resp.status_code == HTTPStatus.OK:
                # 提取embedding向量（假设是单个图片）
                embedding_vector = resp.output['embeddings'][0]['embedding']
                
                embedding_data = {
                    "embedding": embedding_vector,
                    "model": model,
                    "status_code": resp.status_code,
                    "request_id": getattr(resp, "request_id", ""),
                    "usage": resp.usage if hasattr(resp, 'usage') else None
                }
                
                # 记录请求和响应日志
                request_data = {
                    "model": model,
                    "input": f"image_base64_length: {len(image_base64)}",
                    "sdk": "dashscope"
                }
                
                response_content = f"embedding_dimension: {len(embedding_vector)}, model: {model}, status: {resp.status_code}"
                self._log_request_response(request_data, response_content, "get_image_embedding")
                
                return embedding_data
            else:
                error_msg = f"Image embedding failed: status_code={resp.status_code}, code={getattr(resp, 'code', '')}, message={getattr(resp, 'message', '')}"
                logger.error(error_msg)
                raise QwenVLError(error_msg)
            
        except Exception as e:
            logger.error(f"获取图片embedding失败: {e}")
            raise self._handle_api_error(e)
    
    def _crop_face_from_base64(self, image_base64: str, face_rect: Dict[str, int]) -> str:
        """
        从 base64 图片中裁剪人脸区域
        
        Args:
            image_base64: 图片的base64编码
            face_rect: 人脸矩形区域 {"x": int, "y": int, "width": int, "height": int}
            
        Returns:
            裁剪后的人脸图片base64编码
        """
        try:
            # 解码base64图片
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))
            
            # 获取裁剪区域
            x = face_rect["x"]
            y = face_rect["y"]
            width = face_rect["width"]
            height = face_rect["height"]
            
            # 裁剪人脸区域（扩大一些边界以包含更多上下文）
            padding = min(width, height) * 0.2  # 20%的填充
            x1 = max(0, int(x - padding))
            y1 = max(0, int(y - padding))
            x2 = min(image.width, int(x + width + padding))
            y2 = min(image.height, int(y + height + padding))
            
            # 裁剪图片
            face_image = image.crop((x1, y1, x2, y2))
            
            # 转换为base64
            buffered = io.BytesIO()
            face_image.save(buffered, format="JPEG", quality=95)
            face_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            logger.debug(f"裁剪人脸成功: {x1},{y1} -> {x2},{y2}")
            return face_base64
            
        except Exception as e:
            logger.error(f"裁剪人脸失败: {e}")
            raise
    
    @retry_with_backoff()
    def get_face_embedding(self, image_base64: str, face_rect: Dict[str, int], model: str = "multimodal-embedding-v1") -> Dict[str, Any]:
        """
        获取人脸区域的embedding向量
        
        Args:
            image_base64: 原图片的base64编码
            face_rect: 人脸矩形区域
            model: embedding模型名称
            
        Returns:
            包含embedding向量的响应字典
        """
        try:
            # 裁剪人脸图片
            face_base64 = self._crop_face_from_base64(image_base64, face_rect)
            
            # 获取人脸图片的embedding（使用multimodal embedding）
            return self.get_image_embedding(face_base64, model)
            
        except Exception as e:
            logger.error(f"获取人脸embedding失败: {e}")
            raise self._handle_api_error(e)
    
    
    def get_client_info(self) -> Dict[str, Any]:
        """获取客户端信息"""
        return {
            "model": self.qwen_config.model,
            "base_url": self.qwen_config.base_url,
            "max_tokens": self.qwen_config.max_tokens,
            "temperature": self.qwen_config.temperature,
            "timeout": self.qwen_config.timeout,
            "supported_embedding_models": {
                "text": "text-embedding-v4 (via dashscope SDK)",
                "image": "multimodal-embedding-v1 (via dashscope SDK)"
            },
            "dashscope_api_configured": dashscope.api_key is not None
        }