"""
Qwen API 统一客户端
"""
import logging
import json
import re
import sys
from typing import Dict, Any, List, Optional
from openai import OpenAI

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
        from config import config as default_config
        
        self.qwen_config = qwen_config or default_config.qwen_vl
        self.retry_config = retry_config or default_config.retry
        self.enable_logging = enable_logging
        
        # 验证API密钥
        if not self.qwen_config.api_key:
            raise ValueError("DASHSCOPE_API_KEY环境变量未设置或为空")
        
        # 初始化OpenAI客户端（兼容模式）
        self.client = OpenAI(
            api_key=self.qwen_config.api_key,
            base_url=self.qwen_config.base_url
        )
        
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
    
    def get_client_info(self) -> Dict[str, Any]:
        """获取客户端信息"""
        return {
            "model": self.qwen_config.model,
            "base_url": self.qwen_config.base_url,
            "max_tokens": self.qwen_config.max_tokens,
            "temperature": self.qwen_config.temperature,
            "timeout": self.qwen_config.timeout
        }