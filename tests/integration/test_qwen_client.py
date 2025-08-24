"""
QwenClient 真实API集成测试
"""
import json
import os
from datetime import datetime

import numpy as np
import pytest

from clients.qwen_client import (
    QwenClient,
    QwenVLAuthError,
    QwenVLError,
    QwenVLRateLimitError,
    QwenVLServiceError,
)
from config import QwenVLConfig
from tests.test_data import get_test_image_base64

pytestmark = pytest.mark.skipif(
    os.getenv("USE_REAL_API", "false").lower() != "true",
    reason="需要设置 USE_REAL_API=true 才能运行真实API测试",
)


class TestQwenClientRealAPI:
    """QwenClient 真实API集成测试类"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """测试设置"""
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            pytest.skip("需要设置 DASHSCOPE_API_KEY 环境变量")

        self.config = QwenVLConfig(api_key=api_key)
        self.client = QwenClient(qwen_config=self.config)

    @pytest.mark.integration
    def test_client_initialization(self):
        """测试客户端初始化"""
        assert self.client is not None
        assert self.client.qwen_config.api_key == os.getenv("DASHSCOPE_API_KEY")

        client_info = self.client.get_client_info()
        assert "model" in client_info
        assert "base_url" in client_info
        assert client_info["model"] != ""
        assert (
            client_info["base_url"]
            == "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        print(f"✅ 客户端信息: {json.dumps(client_info, indent=2, ensure_ascii=False)}")

    @pytest.mark.integration
    def test_chat_with_text_basic(self):
        """测试基本文本聊天功能"""
        result = self.client.chat_with_text(
            user_prompt="你好，请简单介绍一下你自己", system_prompt="你是一个友好的AI助手"
        )

        assert isinstance(result, str)
        assert len(result) > 0
        print(f"✅ 文本聊天结果: {result}")

    @pytest.mark.integration
    def test_chat_with_text_without_system_prompt(self):
        """测试不带系统提示词的文本聊天"""
        result = self.client.chat_with_text(user_prompt="请用一句话回答：什么是人工智能？")

        assert isinstance(result, str)
        assert len(result) > 0
        print(f"✅ 无系统提示词文本聊天结果: {result}")

    @pytest.mark.integration
    def test_chat_with_image_basic(self):
        """测试基本图片聊天功能"""
        image_base64 = get_test_image_base64()

        result = self.client.chat_with_image(
            image_base64=image_base64,
            user_prompt="请描述这张图片的主要内容",
            system_prompt="你是一个专业的图像分析助手",
        )

        assert isinstance(result, str)
        assert len(result) > 0
        print(f"✅ 图片聊天结果: {result}")

    @pytest.mark.integration
    def test_chat_with_image_without_system_prompt(self):
        """测试不带系统提示词的图片聊天"""
        image_base64 = get_test_image_base64()

        result = self.client.chat_with_image(
            image_base64=image_base64, user_prompt="这张图片显示的是什么？"
        )

        assert isinstance(result, str)
        assert len(result) > 0
        print(f"✅ 无系统提示词图片聊天结果: {result}")

    @pytest.mark.integration
    def test_chat_with_custom_parameters(self):
        """测试自定义参数的聊天"""
        image_base64 = get_test_image_base64()

        # 测试低温度值（更确定性）
        result_low_temp = self.client.chat_with_image(
            image_base64=image_base64,
            user_prompt="请用一个词描述这张图片",
            temperature=0.1,
            max_tokens=50,
        )

        # 测试高温度值（更随机）
        result_high_temp = self.client.chat_with_image(
            image_base64=image_base64,
            user_prompt="请用一个词描述这张图片",
            temperature=0.9,
            max_tokens=50,
        )

        assert isinstance(result_low_temp, str)
        assert isinstance(result_high_temp, str)
        assert len(result_low_temp) > 0
        assert len(result_high_temp) > 0

        print(f"✅ 低温度结果: {result_low_temp}")
        print(f"✅ 高温度结果: {result_high_temp}")

    @pytest.mark.integration
    def test_structured_response(self):
        """测试结构化响应（JSON格式）"""
        image_base64 = get_test_image_base64()

        prompt = """请分析这张图片，并以JSON格式返回以下信息：
{
    "scene_type": "场景类型",
    "main_objects": ["主要物体1", "主要物体2"],
    "description": "详细描述",
    "is_outdoor": "是否户外场景"
}

请只返回JSON格式的结果。"""

        result = self.client.chat_with_image(
            image_base64=image_base64,
            user_prompt=prompt,
            system_prompt="你是专业的图像分析助手，请按要求返回JSON格式结果。",
            temperature=0.3,
        )

        assert isinstance(result, str)
        assert len(result) > 0

        # 尝试解析JSON（不强制要求成功，因为模型可能返回非严格JSON）
        try:
            parsed_result = json.loads(result)
            print(
                f"✅ 结构化分析成功: {json.dumps(parsed_result, indent=2, ensure_ascii=False)}"
            )
        except json.JSONDecodeError:
            print(f"⚠️ JSON解析失败，但获得了响应: {result}")

    @pytest.mark.integration
    def test_error_handling_invalid_api_key(self):
        """测试无效API密钥的错误处理"""
        invalid_config = QwenVLConfig(api_key="invalid_key_12345")
        invalid_client = QwenClient(qwen_config=invalid_config)

        with pytest.raises(QwenVLAuthError):
            invalid_client.chat_with_text("测试消息")

    @pytest.mark.integration
    def test_multiple_consecutive_calls(self):
        """测试连续多次API调用"""
        prompts = ["请简单介绍人工智能", "什么是机器学习？", "深度学习的基本概念是什么？"]

        results = []
        for prompt in prompts:
            result = self.client.chat_with_text(
                user_prompt=prompt, system_prompt="你是AI教育助手，请简洁回答", max_tokens=100
            )
            results.append(result)
            assert isinstance(result, str)
            assert len(result) > 0

        # 验证所有调用都成功
        assert len(results) == len(prompts)
        print(f"✅ 连续调用成功，共 {len(results)} 次")

        for i, (prompt, result) in enumerate(zip(prompts, results), 1):
            print(f"  {i}. {prompt[:20]}... -> {result[:50]}...")

    @pytest.mark.integration
    def test_long_conversation_simulation(self):
        """测试长对话模拟（多次独立调用）"""
        image_base64 = get_test_image_base64()

        conversation_steps = [
            {"prompt": "请描述这张图片", "system": "你是图像分析助手"},
            {"prompt": "这张图片的颜色主要是什么？", "system": "你是色彩分析专家"},
            {"prompt": "如果要给这张图片起个标题，你会起什么？", "system": "你是创意写作助手"},
        ]

        results = []
        for i, step in enumerate(conversation_steps):
            result = self.client.chat_with_image(
                image_base64=image_base64,
                user_prompt=step["prompt"],
                system_prompt=step["system"],
                temperature=0.7,
            )

            results.append(result)
            assert isinstance(result, str)
            assert len(result) > 0

            print(f"✅ 对话步骤 {i+1}: {step['prompt']} -> {result[:80]}...")

        assert len(results) == len(conversation_steps)

    @pytest.mark.integration
    def test_config_parameter_variations(self):
        """测试不同配置参数的效果"""
        # 测试不同模型配置（如果支持）
        configs = [
            {"temperature": 0.1, "max_tokens": 50, "description": "低温度、短回复"},
            {"temperature": 0.8, "max_tokens": 200, "description": "高温度、长回复"},
        ]

        prompt = "请用一句话描述AI的作用"

        for config in configs:
            result = self.client.chat_with_text(
                user_prompt=prompt,
                temperature=config["temperature"],
                max_tokens=config["max_tokens"],
            )

            assert isinstance(result, str)
            assert len(result) > 0

            print(f"✅ {config['description']}: {result}")

    @pytest.mark.integration
    def test_edge_cases(self):
        """测试边界情况"""
        # 测试空白提示词（应该有合理的默认处理）
        try:
            result = self.client.chat_with_text(user_prompt="")
            print(f"✅ 空白提示词处理: {result[:50]}...")
        except Exception as e:
            print(f"⚠️ 空白提示词引发异常（预期行为）: {str(e)[:50]}...")

        # 测试极长提示词（测试token限制）
        long_prompt = "请分析" + "非常" * 100 + "详细的图片内容"
        image_base64 = get_test_image_base64()

        try:
            result = self.client.chat_with_image(
                image_base64=image_base64, user_prompt=long_prompt, max_tokens=100
            )
            assert isinstance(result, str)
            print(f"✅ 长提示词处理成功: {len(long_prompt)} 字符 -> {result[:50]}...")
        except Exception as e:
            print(f"⚠️ 长提示词引发异常: {str(e)[:50]}...")

    @pytest.mark.integration
    def test_concurrent_safety(self):
        """测试客户端并发安全性"""
        import threading
        import time

        results = []
        errors = []

        def make_api_call(thread_id):
            try:
                result = self.client.chat_with_text(
                    user_prompt=f"线程 {thread_id} 的测试消息", max_tokens=50
                )
                results.append((thread_id, result))
            except Exception as e:
                errors.append((thread_id, str(e)))

        # 启动多个线程
        threads = []
        for i in range(3):
            thread = threading.Thread(target=make_api_call, args=(i,))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 验证结果
        print(f"✅ 并发测试完成: {len(results)} 成功, {len(errors)} 错误")

        for thread_id, result in results:
            print(f"  线程 {thread_id}: {result[:50]}...")

        for thread_id, error in errors:
            print(f"  线程 {thread_id} 错误: {error[:50]}...")

        # 至少应该有一些成功的调用
        assert len(results) > 0, "应该至少有一些成功的并发调用"

    @pytest.mark.integration
    def test_text_embedding_basic(self):
        """测试基本文本embedding功能"""
        test_text = "这是一段测试文本，用于验证文本embedding功能"

        result = self.client.get_text_embedding(test_text)

        assert isinstance(result, dict)
        assert "embedding" in result
        assert "model" in result
        assert isinstance(result["embedding"], list)
        assert len(result["embedding"]) > 0

        print(f"✅ 文本embedding成功: 维度={len(result['embedding'])}, 模型={result['model']}")

    @pytest.mark.integration
    def test_image_embedding_basic(self):
        """测试基本图片embedding功能"""
        image_base64 = get_test_image_base64()

        result = self.client.get_image_embedding(image_base64)

        assert isinstance(result, dict)
        assert "embedding" in result
        assert "model" in result
        assert isinstance(result["embedding"], list)
        assert len(result["embedding"]) > 0

        print(f"✅ 图片embedding成功: 维度={len(result['embedding'])}, 模型={result['model']}")

    @pytest.mark.integration
    def test_dashscope_text_embedding_basic(self):
        """测试dashscope文本embedding基本功能"""
        test_text = "美丽的风景照片，包含蓝天白云和绿色的草地"

        result = self.client.get_text_embedding(test_text)

        assert isinstance(result, dict)
        assert "embedding" in result
        assert "model" in result
        assert isinstance(result["embedding"], list)
        assert len(result["embedding"]) > 0
        assert result["model"] == "text-embedding-v4"

        print(
            f"✅ Dashscope文本embedding成功: 维度={len(result['embedding'])}, 模型={result['model']}"
        )

    @pytest.mark.integration
    def test_dashscope_image_embedding_basic(self):
        """测试dashscope图片embedding基本功能"""
        image_base64 = get_test_image_base64()

        result = self.client.get_image_embedding(image_base64)

        assert isinstance(result, dict)
        assert "embedding" in result
        assert "model" in result
        assert isinstance(result["embedding"], list)
        assert len(result["embedding"]) > 0
        assert result["model"] == "multimodal-embedding-v1"

        print(
            f"✅ Dashscope图片embedding成功: 维度={len(result['embedding'])}, 模型={result['model']}"
        )

    @pytest.mark.integration
    def test_embedding_dimension_consistency(self):
        """测试文本和图片embedding维度一致性"""
        image_base64 = get_test_image_base64()
        test_text = "这张图片展示了美丽的风景"

        # 测试文本embedding
        text_result = self.client.get_text_embedding(test_text)

        # 测试图片embedding
        image_result = self.client.get_image_embedding(image_base64)

        assert isinstance(text_result, dict)
        assert isinstance(image_result, dict)
        assert "embedding" in text_result
        assert "embedding" in image_result

        # 检查维度一致性
        text_dim = len(text_result["embedding"])
        image_dim = len(image_result["embedding"])

        print(f"✅ 文本embedding维度: {text_dim}, 图片embedding维度: {image_dim}")
        assert text_dim == image_dim, f"文本和图片embedding维度不一致: {text_dim} vs {image_dim}"

    @pytest.mark.integration
    def test_face_embedding_with_mock_face_rect(self):
        """测试人脸embedding（使用模拟人脸区域）"""
        image_base64 = get_test_image_base64()

        # 模拟人脸区域（假设图片中央有一个人脸）
        mock_face_rect = {"x": 100, "y": 50, "width": 150, "height": 200}

        try:
            result = self.client.get_face_embedding(image_base64, mock_face_rect)

            assert isinstance(result, dict)
            assert "embedding" in result
            assert "model" in result
            assert isinstance(result["embedding"], list)
            assert len(result["embedding"]) > 0

            print(
                f"✅ 人脸embedding成功: 维度={len(result['embedding'])}, 模型={result['model']}"
            )

        except Exception as e:
            print(f"⚠️ 人脸embedding测试失败（可能是图片格式问题）: {str(e)[:100]}...")
            # 不强制要求成功，因为测试图片可能不包含人脸

    @pytest.mark.integration
    def test_embedding_consistency(self):
        """测试embedding一致性（相同输入应得到相同输出）"""
        test_text = "测试embedding一致性的文本"

        # 多次调用相同文本
        result1 = self.client.get_text_embedding(test_text)
        result2 = self.client.get_text_embedding(test_text)

        # 检查向量维度一致
        assert len(result1["embedding"]) == len(result2["embedding"])

        # 检查向量值的相似性（应该非常接近或相同）
        import numpy as np

        vec1 = np.array(result1["embedding"])
        vec2 = np.array(result2["embedding"])
        cosine_similarity = np.dot(vec1, vec2) / (
            np.linalg.norm(vec1) * np.linalg.norm(vec2)
        )

        print(f"✅ Embedding一致性测试: 相似度={cosine_similarity:.6f}")

        # 相似度应该很高（接近1.0）
        assert cosine_similarity > 0.99, f"相同文本的embedding相似度过低: {cosine_similarity}"

    @pytest.mark.integration
    def test_json_data_embedding_simulation(self):
        """测试JSON数据embedding（模拟）"""
        # 模拟从ImageProcessor传来的JSON数据
        json_data = {
            "path": "/test/path/image.jpg",
            "unique_id": "test_img_001",
            "description": "一张美丽的风景照片，包含蓝天白云和绿地",
            "is_landscape": True,
            "has_person": False,
            "processing_status": "success",
            "last_processed": datetime.now().isoformat(),
        }

        # 提取描述文本进行embedding
        description = json_data["description"]
        result = self.client.get_text_embedding(description)

        assert isinstance(result, dict)
        assert "embedding" in result
        assert len(result["embedding"]) > 0

        print(f"✅ JSON数据描述embedding成功: 维度={len(result['embedding'])}")
        print(f"    处理的JSON数据: {json_data['unique_id']} - {description[:50]}...")

    @pytest.mark.integration
    def test_batch_embedding_simulation(self):
        """测试批量embedding（模拟）"""
        test_texts = ["海边的日落景色", "城市夜景灯火通明", "森林中的小溪", "雪山的壮丽景象"]

        results = []
        for text in test_texts:
            result = self.client.get_text_embedding(text)
            results.append(result)
            assert isinstance(result, dict)
            assert "embedding" in result
            assert len(result["embedding"]) > 0

        print(f"✅ 批量embedding成功: 处理了{len(results)}个文本")

        # 验证所有embedding维度一致
        dimensions = [len(r["embedding"]) for r in results]
        assert len(set(dimensions)) == 1, "所有embedding维度应该一致"

        print(f"    所有embedding维度一致: {dimensions[0]}")

    @pytest.mark.integration
    def test_embedding_error_handling(self):
        """测试embedding错误处理"""
        # 测试空文本
        try:
            result = self.client.get_text_embedding("")
            print(f"⚠️ 空文本embedding成功（意外）: {len(result['embedding'])}")
        except Exception as e:
            print(f"✅ 空文本embedding正确抛出异常: {str(e)[:50]}...")

        # 测试无效图片base64数据
        try:
            result = self.client.get_image_embedding("invalid_base64_data")
            print(f"⚠️ 无效图片base64 embedding成功（意外）")
        except Exception as e:
            print(f"✅ 无效图片base64 embedding正确抛出异常: {str(e)[:50]}...")
