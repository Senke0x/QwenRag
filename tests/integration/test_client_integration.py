"""
客户端集成测试
测试 QwenClient 和 PromptManager 的真实API集成使用
"""
import pytest
import os
import base64
from PIL import Image
import io

from clients.qwen_client import QwenClient
from clients.prompt_manager import PromptManager, PromptType
from processors.image_processor import ImageProcessor
from config import QwenVLConfig
from tests.test_data import get_test_image, get_test_image_base64, get_test_portrait


pytestmark = pytest.mark.skipif(
    os.getenv("USE_REAL_API", "false").lower() != "true",
    reason="需要设置 USE_REAL_API=true 才能运行真实API测试"
)


class TestClientIntegration:
    """客户端集成测试类"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """测试设置"""
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            pytest.skip("需要设置 DASHSCOPE_API_KEY 环境变量")
        
        self.config = QwenVLConfig(api_key=api_key)
    
    @pytest.mark.integration
    def test_qwen_client_prompt_manager_integration(self):
        """测试 QwenClient 和 PromptManager 的真实API基本集成"""
        # 创建组件
        qwen_client = QwenClient(qwen_config=self.config)
        prompt_manager = PromptManager()
        
        # 获取提示词
        system_prompt = prompt_manager.get_system_prompt(PromptType.IMAGE_ANALYSIS)
        user_prompt = prompt_manager.get_user_prompt(PromptType.IMAGE_ANALYSIS)
        
        # 调用真实API
        result = qwen_client.chat_with_image(
            image_base64=get_test_image_base64(),
            user_prompt=user_prompt,
            system_prompt=system_prompt
        )
        
        # 验证结果
        assert isinstance(result, str)
        assert len(result) > 0
        
        print(f"✅ 集成测试结果: {result[:100]}...")
        
        # 验证提示词包含预期内容
        assert '图像分析助手' in system_prompt
        assert 'is_snap' in user_prompt
        assert 'is_landscape' in user_prompt
        assert 'description' in user_prompt
    
    @pytest.mark.integration
    def test_image_processor_with_custom_clients(self):
        """测试 ImageProcessor 使用自定义客户端和提示词管理器的真实API集成"""
        # 创建自定义组件
        qwen_client = QwenClient(qwen_config=self.config)
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
        
        # 使用真实图片进行处理
        test_image_path = get_test_image()
        result = processor.analyze_image(test_image_path)
        
        # 验证返回格式
        assert isinstance(result, dict)
        assert 'is_snap' in result
        assert 'has_person' in result
        assert 'description' in result
        assert 'face_rects' in result
        
        # 验证数据类型
        assert isinstance(result['is_snap'], bool)
        assert isinstance(result['has_person'], bool)
        assert isinstance(result['description'], str)
        assert isinstance(result['face_rects'], list)
        
        print(f"✅ 自定义客户端集成测试:")
        print(f"  - 描述: {result['description'][:100]}...")
        print(f"  - 特征: 截图={result['is_snap']}, 人物={result['has_person']}")
    
    @pytest.mark.integration
    def test_multiple_prompt_types_integration(self):
        """测试多种提示词类型的真实API集成使用"""
        qwen_client = QwenClient(qwen_config=self.config)
        prompt_manager = PromptManager()
        
        # 测试不同类型的提示词
        test_cases = [
            {
                'prompt_type': PromptType.IMAGE_ANALYSIS,
                'keywords': ['is_snap', 'is_landscape', 'description', 'has_person']
            },
            {
                'prompt_type': PromptType.FACE_DETECTION, 
                'keywords': ['faces', 'bbox', 'confidence']
            },
            {
                'prompt_type': PromptType.SCENE_CLASSIFICATION,
                'keywords': ['scene_type', 'categories', 'objects']
            }
        ]
        
        for test_case in test_cases:
            # 获取提示词
            prompt = prompt_manager.get_prompt(test_case['prompt_type'])
            
            # 验证提示词包含期望的关键词
            for keyword in test_case['keywords']:
                assert keyword in prompt['user'], f"Keyword '{keyword}' not found in {test_case['prompt_type'].value} prompt"
            
            # 调用真实API
            result = qwen_client.chat_with_image(
                image_base64=get_test_image_base64(),
                user_prompt=prompt['user'],
                system_prompt=prompt['system'],
                temperature=0.3  # 使用较低温度获得更一致的结果
            )
            
            # 验证返回结果
            assert isinstance(result, str)
            assert len(result) > 0
            
            print(f"✅ {test_case['prompt_type'].value} 测试:")
            print(f"  - 结果: {result[:100]}...")
    
    @pytest.mark.integration
    def test_error_handling_integration(self):
        """测试错误处理的真实API集成"""
        # 测试无效API key的错误处理
        from clients.qwen_client import QwenVLAuthError
        
        invalid_config = QwenVLConfig(api_key="invalid_key_123")
        invalid_client = QwenClient(qwen_config=invalid_config)
        prompt_manager = PromptManager()
        processor = ImageProcessor(
            qwen_client=invalid_client,
            prompt_manager=prompt_manager
        )
        
        # 测试错误传播
        test_image_path = get_test_image()
        result = processor.process_image(test_image_path)
        
        # 验证错误被正确处理
        from schemas.data_models import ProcessingStatus
        assert result.processing_status == ProcessingStatus.FAILED
        assert result.error_message != ""
        
        print(f"✅ 错误处理集成测试:")
        print(f"  - 状态: {result.processing_status}")
        print(f"  - 错误信息: {result.error_message[:100]}...")
    
    @pytest.mark.integration
    def test_custom_prompt_workflow(self):
        """测试自定义提示词工作流的真实API集成"""
        qwen_client = QwenClient(qwen_config=self.config)
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
        
        # 调用真实API
        result = qwen_client.chat_with_image(
            image_base64=get_test_image_base64(),
            user_prompt=formatted_user,
            system_prompt=formatted_system
        )
        
        # 验证结果
        assert isinstance(result, str)
        assert len(result) > 0
        
        # 验证参数化结果正确应用
        assert "专业的视觉分析师" in formatted_system
        assert "详细分析这张图片的色彩构成特征" in formatted_user
        
        print(f"✅ 自定义提示词工作流测试:")
        print(f"  - 系统提示词: {formatted_system}")
        print(f"  - 用户提示词: {formatted_user}")
        print(f"  - 分析结果: {result[:100]}...")
    
    @pytest.mark.integration
    def test_client_reuse_across_components(self):
        """测试客户端在多个组件间的复用"""
        # 创建共享的客户端和提示词管理器
        shared_qwen_client = QwenClient(qwen_config=self.config)
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
        
        # 验证独立调用都能正常工作
        test_image_path = get_test_image()
        
        result1 = processor1.analyze_image(test_image_path)
        result2 = processor2.analyze_image(test_image_path)
        
        # 验证每个组件都能正常工作
        assert isinstance(result1, dict)
        assert isinstance(result2, dict)
        assert 'description' in result1
        assert 'description' in result2
        
        print(f"✅ 客户端复用测试:")
        print(f"  - 处理器1结果: {result1['description'][:50]}...")
        print(f"  - 处理器2结果: {result2['description'][:50]}...")
        print(f"  - 共享客户端: {processor1.qwen_client is processor2.qwen_client}")
    
    @pytest.mark.integration
    def test_configuration_consistency(self):
        """测试配置的一致性"""
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            pytest.skip("需要设置 DASHSCOPE_API_KEY 环境变量")
            
        config = QwenVLConfig(
            api_key=api_key,
            model="qwen-vl-max-latest",
            temperature=0.8,
            max_tokens=1024,
            timeout=90
        )
        
        qwen_client = QwenClient(qwen_config=config)
        prompt_manager = PromptManager()
        
        processor = ImageProcessor(
            qwen_client=qwen_client,
            prompt_manager=prompt_manager
        )
        
        # 验证配置被正确传递
        client_info = qwen_client.get_client_info()
        
        assert client_info['model'] == "qwen-vl-max-latest"
        assert client_info['temperature'] == 0.8
        assert client_info['max_tokens'] == 1024
        assert client_info['timeout'] == 90
        assert client_info['base_url'] == "https://dashscope.aliyuncs.com/compatible-mode/v1"
        
        print(f"✅ 配置一致性测试:")
        print(f"  - 模型: {client_info['model']}")
        print(f"  - 温度: {client_info['temperature']}")
        print(f"  - 最大token: {client_info['max_tokens']}")
        print(f"  - 超时: {client_info['timeout']}")
    
    @pytest.mark.integration
    def test_simple_image_generation_integration(self):
        """测试简单生成图片的真实API集成"""
        qwen_client = QwenClient(qwen_config=self.config)
        prompt_manager = PromptManager()
        
        # 创建简单的测试图片
        image = Image.new('RGB', (100, 100), color='red')
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        # 测试图片对话
        image_prompt = prompt_manager.get_prompt(PromptType.IMAGE_ANALYSIS)
        image_result = qwen_client.chat_with_image(
            image_base64=image_base64,
            user_prompt=image_prompt['user'],
            system_prompt=image_prompt['system']
        )
        
        assert isinstance(image_result, str)
        assert len(image_result) > 0
        
        print(f"✅ 简单图片生成集成测试:")
        print(f"  - 图片分析结果: {image_result[:100]}...")
    
    @pytest.mark.integration
    def test_thread_safety(self):
        """测试线程安全性（真实API）"""
        import threading
        import time
        
        # 共享的客户端实例
        shared_client = QwenClient(qwen_config=self.config)
        shared_prompt_manager = PromptManager()
        
        results = []
        errors = []
        
        def worker(worker_id):
            try:
                # 进行真实API调用，但限制数量避免过多调用
                result = shared_client.chat_with_text(
                    user_prompt=f"线程 {worker_id} 的简短测试",
                    max_tokens=50
                )
                results.append((worker_id, result))
                
            except Exception as e:
                errors.append((worker_id, str(e)))
        
        # 启动少量线程进行真实API测试
        threads = []
        for i in range(2):  # 只用2个线程避免API限流
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证结果
        print(f"✅ 线程安全测试: {len(results)} 成功, {len(errors)} 错误")
        
        for worker_id, result in results:
            print(f"  线程 {worker_id}: {result[:50]}...")
        
        for worker_id, error in errors:
            print(f"  线程 {worker_id} 错误: {error[:50]}...")
        
        # 至少应该有一些成功的调用
        assert len(results) > 0, "应该至少有一些成功的线程调用"