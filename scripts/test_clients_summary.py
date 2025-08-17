#!/usr/bin/env python3
"""
客户端测试总结脚本
快速验证重构后的客户端功能
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_qwen_client():
    """测试 QwenClient 基本功能"""
    print("🧪 测试 QwenClient...")
    
    try:
        from clients.qwen_client import QwenClient
        from config import QwenVLConfig
        from unittest.mock import patch, Mock
        
        config = QwenVLConfig(api_key="test_key")
        
        with patch('clients.qwen_client.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            # 模拟API响应
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "测试响应"
            mock_client.chat.completions.create.return_value = mock_response
            
            client = QwenClient(qwen_config=config)
            
            # 测试基本功能
            info = client.get_client_info()
            assert info['model'] == 'qwen-vl-max-latest'
            
            # 测试文本聊天
            result = client.chat_with_text("测试消息")
            assert result == "测试响应"
            
            # 测试图片聊天 - 使用真实图片数据
            from tests.test_data import get_test_image_base64
            real_image_base64 = get_test_image_base64()
            result = client.chat_with_image(real_image_base64, "分析图片")
            assert result == "测试响应"
            
            print("  ✅ QwenClient 测试通过")
            return True
            
    except Exception as e:
        print(f"  ❌ QwenClient 测试失败: {e}")
        return False


def test_prompt_manager():
    """测试 PromptManager 基本功能"""
    print("🧪 测试 PromptManager...")
    
    try:
        from clients.prompt_manager import PromptManager, PromptType
        
        pm = PromptManager()
        
        # 测试基本功能
        types = pm.list_prompt_types()
        assert len(types) >= 4
        
        # 测试获取提示词
        prompt = pm.get_prompt(PromptType.IMAGE_ANALYSIS)
        assert 'system' in prompt
        assert 'user' in prompt
        assert '图像分析助手' in prompt['system']
        
        # 测试参数化提示词
        prompt_with_params = pm.get_prompt(
            PromptType.TEXT_GENERATION,
            user_request="测试请求"
        )
        assert "测试请求" in prompt_with_params['user']
        
        # 测试自定义提示词
        pm.add_prompt(
            "test_custom",
            "测试系统提示",
            "测试用户提示: {param}"
        )
        
        custom_prompt = pm.get_prompt("test_custom", param="参数值")
        assert "参数值" in custom_prompt['user']
        
        print("  ✅ PromptManager 测试通过")
        return True
        
    except Exception as e:
        print(f"  ❌ PromptManager 测试失败: {e}")
        return False


def test_integration():
    """测试集成功能"""
    print("🧪 测试集成功能...")
    
    try:
        from clients.qwen_client import QwenClient
        from clients.prompt_manager import PromptManager, PromptType
        from processors.image_processor import ImageProcessor
        from config import QwenVLConfig
        from unittest.mock import patch, Mock
        
        config = QwenVLConfig(api_key="test_key")
        
        with patch('clients.qwen_client.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            # 模拟API响应
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = '{"is_snap": false, "is_landscape": true, "description": "测试图片", "has_person": false, "face_rects": []}'
            mock_client.chat.completions.create.return_value = mock_response
            
            # 创建组件
            qwen_client = QwenClient(qwen_config=config)
            prompt_manager = PromptManager()
            processor = ImageProcessor(
                qwen_client=qwen_client,
                prompt_manager=prompt_manager
            )
            
            # 验证组件引用
            assert processor.qwen_client is qwen_client
            assert processor.prompt_manager is prompt_manager
            
            # 验证配置一致性
            info = qwen_client.get_client_info()
            assert info['model'] == config.model
            
            print("  ✅ 集成测试通过")
            return True
            
    except Exception as e:
        print(f"  ❌ 集成测试失败: {e}")
        return False


def test_error_handling():
    """测试错误处理"""
    print("🧪 测试错误处理...")
    
    try:
        from clients.qwen_client import QwenClient, QwenVLAuthError
        from clients.prompt_manager import PromptManager
        from config import QwenVLConfig
        from unittest.mock import patch, Mock
        
        # 测试无效API key
        try:
            QwenClient(qwen_config=QwenVLConfig(api_key=""))
            assert False, "应该抛出ValueError"
        except ValueError:
            pass  # 预期的错误
        
        # 测试API错误处理
        config = QwenVLConfig(api_key="test_key")
        
        with patch('clients.qwen_client.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            # 模拟认证错误
            class MockAuthError(Exception):
                def __init__(self, message):
                    super().__init__(message)
                    self.response = type('Response', (), {'status_code': 401})()
            
            mock_client.chat.completions.create.side_effect = MockAuthError("Unauthorized")
            
            client = QwenClient(qwen_config=config)
            
            try:
                client.chat_with_text("测试")
                assert False, "应该抛出QwenVLAuthError"
            except QwenVLAuthError:
                pass  # 预期的错误
        
        # 测试提示词错误
        pm = PromptManager()
        try:
            pm.get_prompt("nonexistent_type")
            assert False, "应该抛出ValueError"
        except ValueError:
            pass  # 预期的错误
        
        print("  ✅ 错误处理测试通过")
        return True
        
    except Exception as e:
        print(f"  ❌ 错误处理测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("🚀 开始客户端功能测试")
    print("=" * 50)
    
    tests = [
        test_qwen_client,
        test_prompt_manager,
        test_integration,
        test_error_handling
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        if test_func():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！客户端重构成功！")
        
        print("\n💡 重构亮点:")
        print("  ✅ 统一的API客户端封装")
        print("  ✅ 集中的提示词管理")
        print("  ✅ 完善的错误处理机制")
        print("  ✅ 组件间松耦合设计")
        print("  ✅ 支持参数化提示词")
        
        return True
    else:
        print(f"❌ 有 {total - passed} 个测试失败")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)