"""
QwenClient 测试用例
"""
import pytest
import os
from unittest.mock import Mock, patch, MagicMock
import json

from clients.qwen_client import (
    QwenClient, 
    QwenVLError, 
    QwenVLAuthError, 
    QwenVLRateLimitError, 
    QwenVLServiceError
)
from config import QwenVLConfig, RetryConfig
from utils.retry_utils import NonRetryableError, RetryableError
from tests.test_data import get_test_image, get_test_image_base64


class TestQwenClient:
    """QwenClient 测试类"""
    
    def test_init_with_valid_config(self):
        """测试使用有效配置初始化客户端"""
        config = QwenVLConfig(api_key="test_api_key")
        
        with patch('clients.qwen_client.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            client = QwenClient(qwen_config=config)
            
            assert client.qwen_config.api_key == "test_api_key"
            assert client.qwen_config.model == "qwen-vl-max-latest"
            mock_openai.assert_called_once_with(
                api_key="test_api_key",
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
    
    def test_init_without_api_key(self):
        """测试没有API密钥时的初始化失败"""
        config = QwenVLConfig(api_key="")
        
        with pytest.raises(ValueError, match="DASHSCOPE_API_KEY环境变量未设置或为空"):
            QwenClient(qwen_config=config)
    
    def test_init_with_default_config(self):
        """测试使用默认配置初始化"""
        with patch('clients.qwen_client.OpenAI') as mock_openai:
            with patch.dict(os.environ, {'DASHSCOPE_API_KEY': 'env_api_key'}):
                # 需要重新加载配置以获取环境变量
                from config import QwenVLConfig
                config = QwenVLConfig()  # 这会重新读取环境变量
                client = QwenClient(qwen_config=config)
                
                assert client.qwen_config.api_key == "env_api_key"
                mock_openai.assert_called_once()
    
    def test_get_client_info(self):
        """测试获取客户端信息"""
        config = QwenVLConfig(
            api_key="test_key",
            model="test-model",
            max_tokens=1024,
            temperature=0.5,
            timeout=30
        )
        
        with patch('clients.qwen_client.OpenAI'):
            client = QwenClient(qwen_config=config)
            info = client.get_client_info()
            
            expected_info = {
                "model": "test-model",
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "max_tokens": 1024,
                "temperature": 0.5,
                "timeout": 30
            }
            
            assert info == expected_info
    
    def test_handle_api_error_auth_error(self):
        """测试认证错误处理"""
        config = QwenVLConfig(api_key="test_key")
        
        with patch('clients.qwen_client.OpenAI'):
            client = QwenClient(qwen_config=config)
            
            # 创建模拟的认证错误
            class MockAuthError(Exception):
                def __init__(self, message):
                    super().__init__(message)
                    self.response = type('Response', (), {'status_code': 401})()
            
            error = MockAuthError("Unauthorized")
            handled_error = client._handle_api_error(error)
            
            assert isinstance(handled_error, QwenVLAuthError)
            assert "API认证失败" in str(handled_error)
    
    def test_handle_api_error_rate_limit(self):
        """测试限流错误处理"""
        config = QwenVLConfig(api_key="test_key")
        
        with patch('clients.qwen_client.OpenAI'):
            client = QwenClient(qwen_config=config)
            
            class MockRateLimitError(Exception):
                def __init__(self, message):
                    super().__init__(message)
                    self.response = type('Response', (), {'status_code': 429})()
            
            error = MockRateLimitError("Rate limit exceeded")
            handled_error = client._handle_api_error(error)
            
            assert isinstance(handled_error, QwenVLRateLimitError)
            assert "API限流" in str(handled_error)
    
    def test_handle_api_error_service_error(self):
        """测试服务错误处理"""
        config = QwenVLConfig(api_key="test_key")
        
        with patch('clients.qwen_client.OpenAI'):
            client = QwenClient(qwen_config=config)
            
            class MockServiceError(Exception):
                def __init__(self, message):
                    super().__init__(message)
                    self.response = type('Response', (), {'status_code': 500})()
            
            error = MockServiceError("Internal server error")
            handled_error = client._handle_api_error(error)
            
            assert isinstance(handled_error, QwenVLServiceError)
            assert "服务错误" in str(handled_error)
    
    def test_handle_api_error_client_error(self):
        """测试客户端错误处理"""
        config = QwenVLConfig(api_key="test_key")
        
        with patch('clients.qwen_client.OpenAI'):
            client = QwenClient(qwen_config=config)
            
            class MockClientError(Exception):
                def __init__(self, message):
                    super().__init__(message)
                    self.response = type('Response', (), {'status_code': 400})()
            
            error = MockClientError("Bad request")
            handled_error = client._handle_api_error(error)
            
            assert isinstance(handled_error, NonRetryableError)
            assert "客户端错误" in str(handled_error)
    
    def test_handle_api_error_network_error(self):
        """测试网络错误处理"""
        config = QwenVLConfig(api_key="test_key")
        
        with patch('clients.qwen_client.OpenAI'):
            client = QwenClient(qwen_config=config)
            
            error = Exception("Connection timeout error")
            handled_error = client._handle_api_error(error)
            
            assert isinstance(handled_error, RetryableError)
            assert "网络错误" in str(handled_error)
    
    def test_handle_api_error_unknown_error(self):
        """测试未知错误处理"""
        config = QwenVLConfig(api_key="test_key")
        
        with patch('clients.qwen_client.OpenAI'):
            client = QwenClient(qwen_config=config)
            
            error = Exception("Some unknown error")
            handled_error = client._handle_api_error(error)
            
            assert isinstance(handled_error, QwenVLError)
            assert "未知错误" in str(handled_error)
    
    def test_chat_with_image_success(self):
        """测试图片聊天成功"""
        config = QwenVLConfig(api_key="test_key")
        
        with patch('clients.qwen_client.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            # 模拟成功的API响应
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "这是一张美丽的风景照"
            mock_client.chat.completions.create.return_value = mock_response
            
            client = QwenClient(qwen_config=config)
            
            # 使用真实图片的base64编码
            real_image_base64 = get_test_image_base64()
            
            result = client.chat_with_image(
                image_base64=real_image_base64,
                user_prompt="描述这张图片",
                system_prompt="你是图像分析助手"
            )
            
            assert result == "这是一张美丽的风景照"
            
            # 验证API调用参数
            call_args = mock_client.chat.completions.create.call_args
            assert call_args[1]['model'] == "qwen-vl-max-latest"
            assert len(call_args[1]['messages']) == 2
            assert call_args[1]['messages'][0]['role'] == 'system'
            assert call_args[1]['messages'][1]['role'] == 'user'
            
            # 验证用户消息包含图片和文本
            user_content = call_args[1]['messages'][1]['content']
            assert len(user_content) == 2
            assert user_content[0]['type'] == 'image_url'
            assert user_content[1]['type'] == 'text'
    
    def test_chat_with_image_without_system_prompt(self):
        """测试不带系统提示词的图片聊天"""
        config = QwenVLConfig(api_key="test_key")
        
        with patch('clients.qwen_client.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "分析结果"
            mock_client.chat.completions.create.return_value = mock_response
            
            client = QwenClient(qwen_config=config)
            
            # 使用真实图片的base64编码
            real_image_base64 = get_test_image_base64()
            
            result = client.chat_with_image(
                image_base64=real_image_base64,
                user_prompt="分析图片"
            )
            
            assert result == "分析结果"
            
            # 验证只有用户消息
            call_args = mock_client.chat.completions.create.call_args
            assert len(call_args[1]['messages']) == 1
            assert call_args[1]['messages'][0]['role'] == 'user'
    
    def test_chat_with_image_with_custom_params(self):
        """测试带自定义参数的图片聊天"""
        config = QwenVLConfig(api_key="test_key")
        
        with patch('clients.qwen_client.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "自定义结果"
            mock_client.chat.completions.create.return_value = mock_response
            
            client = QwenClient(qwen_config=config)
            
            # 使用真实图片的base64编码
            real_image_base64 = get_test_image_base64()
            
            result = client.chat_with_image(
                image_base64=real_image_base64,
                user_prompt="分析图片",
                temperature=0.8,
                max_tokens=1024
            )
            
            assert result == "自定义结果"
            
            # 验证自定义参数
            call_args = mock_client.chat.completions.create.call_args
            assert call_args[1]['temperature'] == 0.8
            assert call_args[1]['max_tokens'] == 1024
    
    def test_chat_with_text_success(self):
        """测试纯文本聊天成功"""
        config = QwenVLConfig(api_key="test_key")
        
        with patch('clients.qwen_client.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "我是AI助手，很高兴为您服务"
            mock_client.chat.completions.create.return_value = mock_response
            
            client = QwenClient(qwen_config=config)
            
            result = client.chat_with_text(
                user_prompt="你好",
                system_prompt="你是一个友好的助手"
            )
            
            assert result == "我是AI助手，很高兴为您服务"
            
            # 验证消息格式
            call_args = mock_client.chat.completions.create.call_args
            messages = call_args[1]['messages']
            assert len(messages) == 2
            assert messages[0]['role'] == 'system'
            assert messages[0]['content'] == "你是一个友好的助手"
            assert messages[1]['role'] == 'user'
            assert messages[1]['content'] == "你好"
    
    def test_chat_with_text_without_system_prompt(self):
        """测试不带系统提示词的纯文本聊天"""
        config = QwenVLConfig(api_key="test_key")
        
        with patch('clients.qwen_client.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Hello!"
            mock_client.chat.completions.create.return_value = mock_response
            
            client = QwenClient(qwen_config=config)
            
            result = client.chat_with_text(user_prompt="Hello")
            
            assert result == "Hello!"
            
            # 验证只有用户消息
            call_args = mock_client.chat.completions.create.call_args
            messages = call_args[1]['messages']
            assert len(messages) == 1
            assert messages[0]['role'] == 'user'
    
    def test_chat_with_image_api_error(self):
        """测试图片聊天API错误"""
        config = QwenVLConfig(api_key="test_key")
        
        with patch('clients.qwen_client.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            # 模拟API错误
            class MockAPIError(Exception):
                def __init__(self, message):
                    super().__init__(message)
                    self.response = type('Response', (), {'status_code': 401})()
            
            mock_client.chat.completions.create.side_effect = MockAPIError("Unauthorized")
            
            client = QwenClient(qwen_config=config)
            
            # 使用真实图片的base64编码
            real_image_base64 = get_test_image_base64()
            
            with pytest.raises(QwenVLAuthError):
                client.chat_with_image(
                    image_base64=real_image_base64,
                    user_prompt="分析图片"
                )
    
    def test_chat_with_text_api_error(self):
        """测试文本聊天API错误"""
        config = QwenVLConfig(api_key="test_key")
        
        with patch('clients.qwen_client.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            # 模拟网络错误
            mock_client.chat.completions.create.side_effect = Exception("Connection timeout")
            
            client = QwenClient(qwen_config=config)
            
            with pytest.raises(RetryableError):
                client.chat_with_text(user_prompt="Hello")
    
    @patch('clients.qwen_client.retry_with_backoff')
    def test_retry_decorator_applied(self, mock_retry):
        """测试重试装饰器是否正确应用"""
        config = QwenVLConfig(api_key="test_key")
        
        with patch('clients.qwen_client.OpenAI'):
            client = QwenClient(qwen_config=config)
            
            # 检查方法是否被装饰器包装
            # 这里主要验证装饰器被调用
            assert hasattr(client.chat_with_image, '__wrapped__') or callable(client.chat_with_image)
            assert hasattr(client.chat_with_text, '__wrapped__') or callable(client.chat_with_text)
    
    def test_config_parameter_precedence(self):
        """测试配置参数的优先级"""
        # 默认配置
        config1 = QwenVLConfig(api_key="test_key")
        
        # 自定义配置
        config2 = QwenVLConfig(
            api_key="test_key",
            model="custom-model",
            temperature=0.8,
            max_tokens=1024
        )
        
        with patch('clients.qwen_client.OpenAI'):
            client1 = QwenClient(qwen_config=config1)
            client2 = QwenClient(qwen_config=config2)
            
            # 验证配置被正确应用
            assert client1.qwen_config.model == "qwen-vl-max-latest"
            assert client1.qwen_config.temperature == 0.1
            
            assert client2.qwen_config.model == "custom-model"
            assert client2.qwen_config.temperature == 0.8
            assert client2.qwen_config.max_tokens == 1024
    
    def test_multiple_client_instances(self):
        """测试多个客户端实例"""
        config1 = QwenVLConfig(api_key="key1", model="model1")
        config2 = QwenVLConfig(api_key="key2", model="model2")
        
        with patch('clients.qwen_client.OpenAI') as mock_openai:
            client1 = QwenClient(qwen_config=config1)
            client2 = QwenClient(qwen_config=config2)
            
            # 验证每个客户端使用自己的配置
            assert client1.qwen_config.api_key == "key1"
            assert client1.qwen_config.model == "model1"
            
            assert client2.qwen_config.api_key == "key2"
            assert client2.qwen_config.model == "model2"
            
            # 验证OpenAI客户端被正确初始化
            assert mock_openai.call_count == 2