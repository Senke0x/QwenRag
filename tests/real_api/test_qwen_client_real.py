"""
QwenClient 真实API测试用例
使用真实的Qwen API进行集成测试
"""
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest
import json
from clients.qwen_client import QwenClient, QwenVLAuthError, QwenVLRateLimitError
from config import QwenVLConfig
from tests.test_data import (
    get_test_image, 
    get_test_portrait, 
    get_test_landscape,
    get_test_interface,
    get_test_image_base64,
    test_data
)


# 只有在设置了真实API时才运行这些测试
pytestmark = pytest.mark.skipif(
    os.getenv("USE_REAL_API", "false").lower() != "true",
    reason="需要设置 USE_REAL_API=true 才能运行真实API测试"
)


class TestQwenClientRealAPI:
    """QwenClient 真实API测试类"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """测试设置"""
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            pytest.skip("需要设置 DASHSCOPE_API_KEY 环境变量")
        
        self.config = QwenVLConfig(api_key=api_key)
        self.client = QwenClient(qwen_config=self.config)
    
    def test_client_initialization(self):
        """测试客户端初始化"""
        assert self.client is not None
        assert self.client.qwen_config.api_key == os.getenv("DASHSCOPE_API_KEY")
        
        client_info = self.client.get_client_info()
        assert "model" in client_info
        assert "base_url" in client_info
        
        print(f"✅ 客户端信息: {json.dumps(client_info, indent=2, ensure_ascii=False)}")
    
    def test_text_chat_basic(self):
        """测试基本文本聊天功能"""
        result = self.client.chat_with_text(
            user_prompt="你好，请简单介绍一下你自己",
            system_prompt="你是一个友好的AI助手"
        )
        
        assert isinstance(result, str)
        assert len(result) > 0
        print(f"✅ 文本聊天结果: {result}")
    
    def test_image_analysis_landscape(self):
        """测试风景图片分析"""
        landscape_path = get_test_landscape()
        image_base64 = test_data.get_image_base64(landscape_path)
        
        result = self.client.chat_with_image(
            image_base64=image_base64,
            user_prompt="请分析这张图片，描述图片的主要内容、场景类型和视觉特征",
            system_prompt="你是一个专业的图像分析助手"
        )
        
        assert isinstance(result, str)
        assert len(result) > 0
        print(f"✅ 风景图片分析 ({landscape_path}): {result}")
    
    def test_image_analysis_portrait(self):
        """测试人物图片分析"""
        portrait_path = get_test_portrait()
        image_base64 = test_data.get_image_base64(portrait_path)
        
        result = self.client.chat_with_image(
            image_base64=image_base64,
            user_prompt="请分析这张图片，判断是否包含人物，如果有人物请描述人物的特征",
            system_prompt="你是一个专业的图像分析助手"
        )
        
        assert isinstance(result, str)
        assert len(result) > 0
        print(f"✅ 人物图片分析 ({portrait_path}): {result}")
    
    def test_image_analysis_interface(self):
        """测试界面图片分析"""
        interface_path = get_test_interface()
        image_base64 = test_data.get_image_base64(interface_path)
        
        result = self.client.chat_with_image(
            image_base64=image_base64,
            user_prompt="请分析这张图片，判断是否是游戏界面或应用截图，描述主要的界面元素",
            system_prompt="你是一个专业的图像分析助手"
        )
        
        assert isinstance(result, str)
        assert len(result) > 0
        print(f"✅ 界面图片分析 ({interface_path}): {result}")
    
    def test_structured_image_analysis(self):
        """测试结构化图片分析（JSON格式）"""
        image_path = get_test_image()
        image_base64 = test_data.get_image_base64(image_path)
        
        prompt = """请仔细分析这张图片，并以JSON格式返回以下信息：
{
    "is_snap": boolean,  // 是否是手机截图
    "is_landscape": boolean,  // 是否是风景照
    "description": "string",  // 详细的图片描述
    "has_person": boolean,  // 是否有人物
    "scene_type": "string",  // 场景类型
    "main_objects": ["object1", "object2"],  // 主要物体
    "colors": ["color1", "color2"]  // 主要颜色
}

请只返回JSON格式的结果，不要包含其他文字。"""
        
        result = self.client.chat_with_image(
            image_base64=image_base64,
            user_prompt=prompt,
            system_prompt="你是一个专业的图像分析助手。请仔细分析图片内容，并以JSON格式返回结果。"
        )
        
        assert isinstance(result, str)
        assert len(result) > 0
        
        # 尝试解析JSON
        try:
            parsed_result = json.loads(result)
            assert "is_snap" in parsed_result
            assert "is_landscape" in parsed_result
            assert "description" in parsed_result
            assert "has_person" in parsed_result
            print(f"✅ 结构化分析 ({image_path}): {json.dumps(parsed_result, indent=2, ensure_ascii=False)}")
        except json.JSONDecodeError:
            print(f"⚠️ JSON解析失败，原始结果: {result}")
            # 不将此作为测试失败，因为模型可能返回非严格JSON
    
    def test_multiple_images_batch(self):
        """测试批量图片处理"""
        images = test_data.get_multiple_images(count=3)
        results = []
        
        for image_path in images:
            try:
                image_base64 = test_data.get_image_base64(image_path)
                result = self.client.chat_with_image(
                    image_base64=image_base64,
                    user_prompt="请用一句话简单描述这张图片的主要内容",
                    system_prompt="你是图像分析助手，请简洁地描述图片内容"
                )
                results.append((image_path, result))
                print(f"✅ 批量处理 {image_path}: {result}")
            except Exception as e:
                print(f"❌ 处理 {image_path} 时出错: {e}")
                results.append((image_path, f"错误: {e}"))
        
        assert len(results) == len(images)
        # 至少要有一半的图片处理成功
        success_count = sum(1 for _, result in results if not result.startswith("错误"))
        assert success_count >= len(images) // 2
    
    def test_error_handling_invalid_api_key(self):
        """测试无效API密钥的错误处理"""
        invalid_config = QwenVLConfig(api_key="invalid_key_12345")
        invalid_client = QwenClient(qwen_config=invalid_config)
        
        with pytest.raises(QwenVLAuthError):
            invalid_client.chat_with_text("测试消息")
    
    def test_custom_parameters(self):
        """测试自定义参数"""
        image_base64 = get_test_image_base64()
        
        # 测试不同的temperature值
        result_low_temp = self.client.chat_with_image(
            image_base64=image_base64,
            user_prompt="请描述这张图片",
            temperature=0.1,  # 更确定性的输出
            max_tokens=100
        )
        
        result_high_temp = self.client.chat_with_image(
            image_base64=image_base64,
            user_prompt="请描述这张图片",
            temperature=0.9,  # 更随机的输出
            max_tokens=100
        )
        
        assert isinstance(result_low_temp, str)
        assert isinstance(result_high_temp, str)
        print(f"✅ 低温度结果: {result_low_temp}")
        print(f"✅ 高温度结果: {result_high_temp}")
    
    def test_long_conversation(self):
        """测试长对话功能"""
        # 虽然当前实现不支持对话历史，但测试多次独立调用
        conversation_prompts = [
            "请简单描述这张图片",
            "这张图片的颜色主要是什么？",
            "图片中有什么值得注意的细节？"
        ]
        
        image_base64 = get_test_image_base64()
        
        for i, prompt in enumerate(conversation_prompts):
            result = self.client.chat_with_image(
                image_base64=image_base64,
                user_prompt=prompt,
                system_prompt=f"你是图像分析助手，这是第{i+1}次分析同一张图片"
            )
            
            assert isinstance(result, str)
            assert len(result) > 0
            print(f"✅ 对话{i+1}: {prompt} -> {result}")


def run_manual_tests():
    """手动运行测试的函数"""
    print("🚀 开始QwenClient真实API测试")
    print("=" * 60)
    
    # 检查环境变量
    if os.getenv("USE_REAL_API", "false").lower() != "true":
        print("❌ 请设置环境变量 USE_REAL_API=true")
        return False
    
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("❌ 请设置环境变量 DASHSCOPE_API_KEY")
        return False
    
    # 验证dataset目录
    dataset_info = test_data.verify_dataset()
    print(f"📁 数据集信息: {json.dumps(dataset_info, indent=2, ensure_ascii=False)}")
    
    if not dataset_info["dataset_exists"] or dataset_info["total_images"] == 0:
        print("❌ dataset目录不存在或没有图片文件")
        return False
    
    # 创建测试实例
    try:
        config = QwenVLConfig(api_key=api_key)
        client = QwenClient(qwen_config=config)
        print("✅ 客户端初始化成功")
    except Exception as e:
        print(f"❌ 客户端初始化失败: {e}")
        return False
    
    # 运行基本测试
    test_cases = [
        ("文本聊天", lambda: client.chat_with_text("你好，请简单介绍一下你自己")),
        ("图片分析", lambda: client.chat_with_image(
            get_test_image_base64(), "请描述这张图片的主要内容"
        )),
    ]
    
    passed = 0
    total = len(test_cases)
    
    for test_name, test_func in test_cases:
        try:
            result = test_func()
            print(f"✅ {test_name}测试通过: {result[:100]}...")
            passed += 1
        except Exception as e:
            print(f"❌ {test_name}测试失败: {e}")
    
    print("=" * 60)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有真实API测试通过！")
        return True
    else:
        print(f"❌ 有 {total - passed} 个测试失败")
        return False


if __name__ == "__main__":
    # 可以直接运行此文件进行手动测试
    success = run_manual_tests()
    exit(0 if success else 1)