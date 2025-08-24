#!/usr/bin/env python3
"""
测试QwenClient日志功能演示
"""
import logging
import sys
from unittest.mock import Mock, patch

from clients.qwen_client import QwenClient
from config import QwenVLConfig
from tests.test_data import get_test_image_base64

# 配置日志输出到控制台
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def test_image_chat_logging():
    """测试图片聊天的日志功能"""
    print("🧪 测试图片聊天日志功能")
    print("=" * 50)

    # 创建配置
    config = QwenVLConfig(api_key="test_key")

    # 使用mock避免真实API调用
    with patch("clients.qwen_client.OpenAI") as mock_openai:
        mock_client = Mock()
        mock_openai.return_value = mock_client

        # 模拟API响应
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[
            0
        ].message.content = '{"is_snap": false, "is_landscape": true, "description": "这是一张美丽的游戏截图，显示了《最后生还者》中的场景", "has_person": true, "face_rects": [[100, 150, 80, 120]]}'
        mock_client.chat.completions.create.return_value = mock_response

        # 创建客户端（启用日志）
        client = QwenClient(qwen_config=config, enable_logging=True)

        # 获取真实图片数据
        image_base64 = get_test_image_base64()

        # 调用API
        result = client.chat_with_image(
            image_base64=image_base64,
            user_prompt="请分析这张图片，并以JSON格式返回结果",
            system_prompt="你是一个专业的图像分析助手",
            temperature=0.7,
            max_tokens=1000,
        )

        print(f"\n✅ API调用成功，返回结果: {result}")

        return result


def test_text_chat_logging():
    """测试文本聊天的日志功能"""
    print("\n🧪 测试文本聊天日志功能")
    print("=" * 50)

    # 创建配置
    config = QwenVLConfig(api_key="test_key")

    # 使用mock避免真实API调用
    with patch("clients.qwen_client.OpenAI") as mock_openai:
        mock_client = Mock()
        mock_openai.return_value = mock_client

        # 模拟API响应
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[
            0
        ].message.content = (
            "你好！我是通义千问，一个由阿里云开发的AI助手。我可以帮助你回答问题、进行对话、协助完成各种任务。有什么我可以帮助你的吗？"
        )
        mock_client.chat.completions.create.return_value = mock_response

        # 创建客户端（启用日志）
        client = QwenClient(qwen_config=config, enable_logging=True)

        # 调用API
        result = client.chat_with_text(
            user_prompt="你好，请简单介绍一下你自己", system_prompt="你是一个友好的AI助手", temperature=0.5
        )

        print(f"\n✅ API调用成功，返回结果: {result}")

        return result


def test_logging_disabled():
    """测试禁用日志的情况"""
    print("\n🧪 测试禁用日志功能")
    print("=" * 50)

    # 创建配置
    config = QwenVLConfig(api_key="test_key")

    # 使用mock避免真实API调用
    with patch("clients.qwen_client.OpenAI") as mock_openai:
        mock_client = Mock()
        mock_openai.return_value = mock_client

        # 模拟API响应
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "无日志输出测试"
        mock_client.chat.completions.create.return_value = mock_response

        # 创建客户端（禁用日志）
        client = QwenClient(qwen_config=config, enable_logging=False)

        # 调用API
        result = client.chat_with_text(user_prompt="这次调用不应该有日志输出", system_prompt="测试系统")

        print(f"✅ API调用成功（无日志输出），返回结果: {result}")

        return result


def main():
    """主测试函数"""
    print("🚀 QwenClient 日志功能测试")
    print("=" * 60)

    try:
        # 测试图片聊天日志
        test_image_chat_logging()

        # 测试文本聊天日志
        test_text_chat_logging()

        # 测试禁用日志
        test_logging_disabled()

        print("\n" + "=" * 60)
        print("🎉 所有日志功能测试完成！")
        print("\n💡 日志功能特点:")
        print("   ✅ 自动替换图片base64数据为mock数据")
        print("   ✅ 格式化显示请求和响应")
        print("   ✅ 支持启用/禁用日志")
        print("   ✅ 显示数据大小信息")
        print("   ✅ 美观的输出格式")

        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
