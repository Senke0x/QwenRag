"""
真实API环境下的日志功能测试
"""
import pytest
import os
import logging
from clients.qwen_client import QwenClient
from config import QwenVLConfig
from tests.test_data import get_test_image, get_test_image_base64

# 配置日志
logging.basicConfig(level=logging.INFO)

# 只有在设置了真实API时才运行这些测试
pytestmark = pytest.mark.skipif(
    os.getenv("USE_REAL_API", "false").lower() != "true",
    reason="需要设置 USE_REAL_API=true 才能运行真实API日志测试"
)


class TestQwenClientLoggingReal:
    """QwenClient 真实API日志测试类"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """测试设置"""
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            pytest.skip("需要设置 DASHSCOPE_API_KEY 环境变量")
        
        self.config = QwenVLConfig(api_key=api_key)
    
    def test_image_chat_with_logging(self):
        """测试图片聊天的日志输出"""
        print("\n🧪 测试真实API图片聊天日志功能")
        
        # 创建启用日志的客户端
        client = QwenClient(qwen_config=self.config, enable_logging=True)
        
        # 使用真实图片数据
        image_base64 = get_test_image_base64()
        
        result = client.chat_with_image(
            image_base64=image_base64,
            user_prompt="请简单描述这张图片的主要内容",
            system_prompt="你是一个图像分析助手，请简洁地描述图片内容",
            temperature=0.7,
            max_tokens=200
        )
        
        assert isinstance(result, str)
        assert len(result) > 0
        print(f"\n✅ 图片分析结果: {result}")
    
    def test_text_chat_with_logging(self):
        """测试文本聊天的日志输出"""
        print("\n🧪 测试真实API文本聊天日志功能")
        
        # 创建启用日志的客户端
        client = QwenClient(qwen_config=self.config, enable_logging=True)
        
        result = client.chat_with_text(
            user_prompt="你好，请用一句话介绍你自己",
            system_prompt="你是一个简洁的AI助手",
            temperature=0.5,
            max_tokens=100
        )
        
        assert isinstance(result, str)
        assert len(result) > 0
        print(f"\n✅ 文本聊天结果: {result}")
    
    def test_chat_without_logging(self):
        """测试禁用日志的情况"""
        print("\n🧪 测试禁用日志功能（应该没有API调用日志）")
        
        # 创建禁用日志的客户端
        client = QwenClient(qwen_config=self.config, enable_logging=False)
        
        result = client.chat_with_text(
            user_prompt="这个测试不会显示API调用日志",
            temperature=0.3,
            max_tokens=50
        )
        
        assert isinstance(result, str)
        assert len(result) > 0
        print(f"\n✅ 禁用日志测试结果: {result}")
    
    def test_structured_response_logging(self):
        """测试结构化响应的日志"""
        print("\n🧪 测试结构化响应日志")
        
        client = QwenClient(qwen_config=self.config, enable_logging=True)
        
        image_base64 = get_test_image_base64()
        
        result = client.chat_with_image(
            image_base64=image_base64,
            user_prompt="""请分析这张图片，以JSON格式返回：
{
    "scene_type": "场景类型",
    "description": "简短描述",
    "has_people": "是否有人物"
}""",
            system_prompt="你是图像分析助手，请按要求返回JSON格式结果",
            temperature=0.3,
            max_tokens=300
        )
        
        assert isinstance(result, str)
        print(f"\n✅ 结构化分析结果: {result}")


def run_logging_demo():
    """手动运行日志演示"""
    print("🚀 QwenClient 真实API日志演示")
    print("=" * 60)
    
    # 检查环境
    if os.getenv("USE_REAL_API", "false").lower() != "true":
        print("❌ 请设置环境变量 USE_REAL_API=true")
        return False
    
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("❌ 请设置环境变量 DASHSCOPE_API_KEY")
        return False
    
    config = QwenVLConfig(api_key=api_key)
    
    print("✅ 环境配置正确")
    print(f"✅ 图片数据大小: {len(get_test_image_base64())} 字符")
    
    # 测试启用日志的客户端
    print("\n📝 创建启用日志的客户端...")
    client_with_log = QwenClient(qwen_config=config, enable_logging=True)
    
    try:
        print("\n🔍 进行图片分析（将显示完整的请求/响应日志）...")
        result = client_with_log.chat_with_image(
            image_base64=get_test_image_base64(),
            user_prompt="请用一句话描述这张图片",
            system_prompt="你是图像分析助手",
            temperature=0.5
        )
        print(f"\n📋 最终结果: {result}")
        
    except Exception as e:
        print(f"❌ API调用失败: {e}")
        return False
    
    # 测试禁用日志的客户端
    print("\n" + "=" * 60)
    print("📝 创建禁用日志的客户端...")
    client_no_log = QwenClient(qwen_config=config, enable_logging=False)
    
    try:
        print("\n🔇 进行文本聊天（不会显示API调用日志）...")
        result = client_no_log.chat_with_text(
            user_prompt="你好",
            temperature=0.3
        )
        print(f"\n📋 最终结果: {result}")
        
    except Exception as e:
        print(f"❌ API调用失败: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 日志功能演示完成！")
    print("\n💡 日志功能说明:")
    print("   📤 REQUEST部分显示发送给API的完整请求")
    print("   📥 RESPONSE部分显示API返回的原始响应")
    print("   🎭 图片base64数据被替换为可读的mock信息")
    print("   📊 显示图片数据的实际字节大小")
    print("   🎛️ 支持通过enable_logging参数控制开关")
    
    return True


if __name__ == "__main__":
    # 可以直接运行此文件进行演示
    success = run_logging_demo()
    exit(0 if success else 1)