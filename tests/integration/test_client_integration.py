"""
客户端集成测试
测试 QwenClient 和 PromptManager 的集成使用
"""
import pytest
import os
from unittest.mock import Mock, patch
import base64
from PIL import Image
import io

from clients.qwen_client import QwenClient
from clients.prompt_manager import PromptManager, PromptType
from processors.image_processor import ImageProcessor
from config import QwenVLConfig


class TestClientIntegration:
    """客户端集成测试类"""
    
    def test_qwen_client_prompt_manager_integration(self):
        """测试 QwenClient 和 PromptManager 的基本集成"""
        config = QwenVLConfig(api_key="test_key")
        
        with patch('clients.qwen_client.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            # 模拟API响应
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = '{"is_snap": false, "is_landscape": true, "description": "美丽的山水风景", "has_person": false, "face_rects": []}'
            mock_client.chat.completions.create.return_value = mock_response
            
            # 创建组件
            qwen_client = QwenClient(qwen_config=config)
            prompt_manager = PromptManager()
            
            # 获取提示词
            system_prompt = prompt_manager.get_system_prompt(PromptType.IMAGE_ANALYSIS)
            user_prompt = prompt_manager.get_user_prompt(PromptType.IMAGE_ANALYSIS)
            
            # 调用客户端
            result = qwen_client.chat_with_image(
                image_base64="fake_base64_image",
                user_prompt=user_prompt,
                system_prompt=system_prompt
            )
            
            assert result == '{"is_snap": false, "is_landscape": true, "description": "美丽的山水风景", "has_person": false, "face_rects": []}'
            
            # 验证API调用参数
            call_args = mock_client.chat.completions.create.call_args
            assert call_args[1]['messages'][0]['content'][0]['text'] == system_prompt
            assert call_args[1]['messages'][1]['content'][1]['text'] == user_prompt
    
    def test_image_processor_with_custom_clients(self):
        """测试 ImageProcessor 使用自定义客户端和提示词管理器"""
        config = QwenVLConfig(api_key="test_key")
        
        with patch('clients.qwen_client.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            # 模拟API响应
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = '{"is_snap": false, "is_landscape": false, "description": "人物肖像", "has_person": true, "face_rects": [[50, 60, 100, 120]]}'
            mock_client.chat.completions.create.return_value = mock_response
            
            # 创建自定义组件
            qwen_client = QwenClient(qwen_config=config)
            prompt_manager = PromptManager()
            
            # 自定义提示词
            prompt_manager.update_prompt(
                PromptType.IMAGE_ANALYSIS,
                system_prompt="自定义图像分析助手",
                user_prompt="请分析这张图片并返回JSON格式的结果"
            )
            
            # 创建图片处理器
            processor = ImageProcessor(
                qwen_client=qwen_client,
                prompt_manager=prompt_manager
            )
            
            # 模拟图片处理
            with patch('utils.image_utils.validate_image_file') as mock_validate:
                with patch('utils.image_utils.image_to_base64') as mock_base64:
                    with patch('utils.image_utils.get_image_info') as mock_info:
                        mock_validate.return_value = (True, None)
                        mock_base64.return_value = "fake_base64"
                        mock_info.return_value = {
                            'unique_id': 'test_id',
                            'timestamp': '2024-01-01T00:00:00'
                        }
                        
                        result = processor.analyze_image("/fake/path/image.jpg")
                        
                        assert result['is_snap'] == False
                        assert result['has_person'] == True
                        assert result['description'] == "人物肖像"
                        assert len(result['face_rects']) == 1
    
    def test_multiple_prompt_types_integration(self):
        """测试多种提示词类型的集成使用"""
        config = QwenVLConfig(api_key="test_key")
        
        with patch('clients.qwen_client.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            qwen_client = QwenClient(qwen_config=config)
            prompt_manager = PromptManager()
            
            # 测试不同类型的提示词
            test_cases = [
                {
                    'prompt_type': PromptType.IMAGE_ANALYSIS,
                    'response': '{"is_snap": true, "is_landscape": false, "description": "手机截图", "has_person": false, "face_rects": []}',
                    'keywords': ['is_snap', 'is_landscape']
                },
                {
                    'prompt_type': PromptType.FACE_DETECTION, 
                    'response': '{"faces": [{"bbox": [10, 20, 50, 60], "confidence": 0.95}], "total_faces": 1}',
                    'keywords': ['faces', 'bbox', 'confidence']
                },
                {
                    'prompt_type': PromptType.SCENE_CLASSIFICATION,
                    'response': '{"scene_type": "outdoor", "categories": ["park"], "objects": ["tree", "bench"]}',
                    'keywords': ['scene_type', 'categories', 'objects']
                }
            ]
            
            for test_case in test_cases:
                # 设置模拟响应
                mock_response = Mock()
                mock_response.choices = [Mock()]
                mock_response.choices[0].message.content = test_case['response']
                mock_client.chat.completions.create.return_value = mock_response
                
                # 获取提示词
                prompt = prompt_manager.get_prompt(test_case['prompt_type'])
                
                # 验证提示词包含期望的关键词
                for keyword in test_case['keywords']:
                    assert keyword in prompt['user'], f"Keyword '{keyword}' not found in {test_case['prompt_type'].value} prompt"
                
                # 调用API
                result = qwen_client.chat_with_image(
                    image_base64="fake_image",
                    user_prompt=prompt['user'],
                    system_prompt=prompt['system']
                )
                
                assert result == test_case['response']
    
    def test_error_handling_integration(self):
        """测试错误处理的集成"""
        config = QwenVLConfig(api_key="test_key")
        
        with patch('clients.qwen_client.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            # 模拟API错误
            class MockAPIError(Exception):
                def __init__(self, message):
                    super().__init__(message)
                    self.response = type('Response', (), {'status_code': 429})()
            
            mock_client.chat.completions.create.side_effect = MockAPIError("Rate limit exceeded")
            
            qwen_client = QwenClient(qwen_config=config)
            prompt_manager = PromptManager()
            processor = ImageProcessor(
                qwen_client=qwen_client,
                prompt_manager=prompt_manager
            )
            
            # 测试错误传播
            with patch('utils.image_utils.validate_image_file') as mock_validate:
                with patch('utils.image_utils.image_to_base64') as mock_base64:
                    mock_validate.return_value = (True, None)
                    mock_base64.return_value = "fake_base64"
                    
                    result = processor.process_image("/fake/path/image.jpg")
                    
                    # 验证错误被正确处理
                    from schemas.data_models import ProcessingStatus
                    assert result.processing_status == ProcessingStatus.FAILED
                    assert "limit" in result.error_message.lower() or "限流" in result.error_message
    
    def test_custom_prompt_workflow(self):
        """测试自定义提示词工作流"""
        config = QwenVLConfig(api_key="test_key")
        
        with patch('clients.qwen_client.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            # 模拟API响应
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "这是一个自定义分析结果"
            mock_client.chat.completions.create.return_value = mock_response
            
            qwen_client = QwenClient(qwen_config=config)
            prompt_manager = PromptManager()
            
            # 添加自定义提示词
            prompt_manager.add_prompt(
                prompt_type="custom_analysis",
                system_prompt="你是一个专业的{domain}分析师",
                user_prompt="请详细分析这张{content_type}的{aspect}特征"
            )
            
            # 使用参数化提示词
            custom_prompts = prompt_manager._prompts["custom_analysis"]
            formatted_system = custom_prompts["system"].format(domain="视觉")
            formatted_user = custom_prompts["user"].format(
                content_type="图片",
                aspect="色彩构成"
            )
            
            result = qwen_client.chat_with_image(
                image_base64="fake_image",
                user_prompt=formatted_user,
                system_prompt=formatted_system
            )
            
            assert result == "这是一个自定义分析结果"
            
            # 验证API调用中的参数化结果
            call_args = mock_client.chat.completions.create.call_args
            system_content = call_args[1]['messages'][0]['content'][0]['text']
            user_content = call_args[1]['messages'][1]['content'][1]['text']
            
            assert "专业的视觉分析师" in system_content
            assert "详细分析这张图片的色彩构成特征" in user_content
    
    def test_client_reuse_across_components(self):
        """测试客户端在多个组件间的复用"""
        config = QwenVLConfig(api_key="test_key")
        
        with patch('clients.qwen_client.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            # 模拟不同的API响应
            responses = [
                '{"analysis": "component1_result"}',
                '{"analysis": "component2_result"}',
                '{"analysis": "component3_result"}'
            ]
            
            response_iter = iter(responses)
            
            def create_mock_response(*args, **kwargs):
                mock_response = Mock()
                mock_response.choices = [Mock()]
                mock_response.choices[0].message.content = next(response_iter)
                return mock_response
            
            mock_client.chat.completions.create.side_effect = create_mock_response
            
            # 创建共享的客户端和提示词管理器
            shared_qwen_client = QwenClient(qwen_config=config)
            shared_prompt_manager = PromptManager()
            
            # 创建多个组件实例
            processor1 = ImageProcessor(
                qwen_client=shared_qwen_client,
                prompt_manager=shared_prompt_manager
            )
            
            processor2 = ImageProcessor(
                qwen_client=shared_qwen_client,
                prompt_manager=shared_prompt_manager
            )
            
            # 验证它们使用相同的客户端实例
            assert processor1.qwen_client is shared_qwen_client
            assert processor2.qwen_client is shared_qwen_client
            assert processor1.prompt_manager is shared_prompt_manager
            assert processor2.prompt_manager is shared_prompt_manager
            
            # 验证独立调用
            with patch('utils.image_utils.validate_image_file', return_value=(True, None)):
                with patch('utils.image_utils.image_to_base64', return_value="fake1"):
                    with patch('utils.image_utils.get_image_info', return_value={'unique_id': 'id1', 'timestamp': 'time1'}):
                        result1 = processor1.analyze_image("/fake1.jpg")
                        
                with patch('utils.image_utils.image_to_base64', return_value="fake2"):
                    with patch('utils.image_utils.get_image_info', return_value={'unique_id': 'id2', 'timestamp': 'time2'}):
                        result2 = processor2.analyze_image("/fake2.jpg")
            
            # 验证每个组件都能正常工作
            assert '"analysis": "component1_result"' in str(result1)
            assert '"analysis": "component2_result"' in str(result2)
            
            # 验证OpenAI客户端只被初始化一次
            assert mock_openai.call_count == 1
    
    def test_configuration_consistency(self):
        """测试配置的一致性"""
        config = QwenVLConfig(
            api_key="test_key",
            model="custom-model",
            temperature=0.8,
            max_tokens=1024,
            timeout=90
        )
        
        with patch('clients.qwen_client.OpenAI') as mock_openai:
            qwen_client = QwenClient(qwen_config=config)
            prompt_manager = PromptManager()
            
            processor = ImageProcessor(
                qwen_client=qwen_client,
                prompt_manager=prompt_manager
            )
            
            # 验证配置被正确传递
            client_info = qwen_client.get_client_info()
            
            assert client_info['model'] == "custom-model"
            assert client_info['temperature'] == 0.8
            assert client_info['max_tokens'] == 1024
            assert client_info['timeout'] == 90
            
            # 验证OpenAI客户端初始化参数
            mock_openai.assert_called_once_with(
                api_key="test_key",
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
    
    @pytest.mark.integration
    def test_real_api_integration(self):
        """真实API集成测试（需要有效的API key）"""
        api_key = os.getenv("DASHSCOPE_API_KEY")
        use_real_api = os.getenv("USE_REAL_API", "false").lower() == "true"
        
        if not use_real_api or not api_key:
            pytest.skip("跳过真实API测试 - 设置USE_REAL_API=true和DASHSCOPE_API_KEY环境变量来启用")
        
        config = QwenVLConfig(api_key=api_key)
        qwen_client = QwenClient(qwen_config=config)
        prompt_manager = PromptManager()
        
        # 创建简单的测试图片
        image = Image.new('RGB', (100, 100), color='red')
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        try:
            # 测试文本对话
            text_result = qwen_client.chat_with_text(
                user_prompt="请简单介绍一下人工智能",
                system_prompt="你是一个知识渊博的AI助手"
            )
            
            assert isinstance(text_result, str)
            assert len(text_result) > 0
            
            # 测试图片对话
            image_prompt = prompt_manager.get_prompt(PromptType.IMAGE_ANALYSIS)
            image_result = qwen_client.chat_with_image(
                image_base64=image_base64,
                user_prompt=image_prompt['user'],
                system_prompt=image_prompt['system']
            )
            
            assert isinstance(image_result, str)
            assert len(image_result) > 0
            
        except Exception as e:
            pytest.fail(f"真实API集成测试失败: {e}")
    
    def test_thread_safety(self):
        """测试线程安全性"""
        import threading
        import time
        
        config = QwenVLConfig(api_key="test_key")
        
        with patch('clients.qwen_client.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            # 模拟API响应
            def create_response(*args, **kwargs):
                mock_response = Mock()
                mock_response.choices = [Mock()]
                mock_response.choices[0].message.content = f"Thread response {threading.current_thread().ident}"
                return mock_response
            
            mock_client.chat.completions.create.side_effect = create_response
            
            # 共享的客户端实例
            shared_client = QwenClient(qwen_config=config)
            shared_prompt_manager = PromptManager()
            
            results = []
            errors = []
            
            def worker():
                try:
                    processor = ImageProcessor(
                        qwen_client=shared_client,
                        prompt_manager=shared_prompt_manager
                    )
                    
                    # 模拟并发API调用
                    for i in range(5):
                        prompt = shared_prompt_manager.get_prompt(PromptType.IMAGE_ANALYSIS)
                        result = shared_client.chat_with_image(
                            image_base64=f"fake_image_{i}",
                            user_prompt=prompt['user'],
                            system_prompt=prompt['system']
                        )
                        results.append(result)
                        time.sleep(0.001)  # 小延迟
                        
                except Exception as e:
                    errors.append(e)
            
            # 启动多个线程
            threads = []
            for _ in range(3):
                thread = threading.Thread(target=worker)
                threads.append(thread)
                thread.start()
            
            # 等待所有线程完成
            for thread in threads:
                thread.join()
            
            # 验证没有错误
            assert len(errors) == 0, f"Thread safety errors: {errors}"
            assert len(results) == 15  # 3个线程 × 5次调用
            
            # 验证所有调用都成功
            for result in results:
                assert "Thread response" in result