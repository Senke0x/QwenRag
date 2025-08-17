"""
Qwen API 统一客户端
"""
import logging
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
        retry_config: Optional[RetryConfig] = None
    ):
        """
        初始化 Qwen 客户端
        
        Args:
            qwen_config: Qwen VL配置
            retry_config: 重试配置
        """
        from config import config as default_config
        
        self.qwen_config = qwen_config or default_config.qwen_vl
        self.retry_config = retry_config or default_config.retry
        
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
            logger.debug(f"Qwen VL API响应: {content}")
            
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
            logger.debug(f"Qwen API响应: {content}")
            
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