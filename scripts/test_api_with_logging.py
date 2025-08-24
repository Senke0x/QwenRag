#!/usr/bin/env python3
"""
简单的API测试，显示request/response日志
"""
import os
import sys
import warnings
from pathlib import Path

# 过滤warnings
warnings.filterwarnings(
    "ignore", message="builtin type SwigPyPacked has no __module__ attribute"
)
warnings.filterwarnings(
    "ignore", message="builtin type SwigPyObject has no __module__ attribute"
)
warnings.filterwarnings(
    "ignore", message="builtin type swigvarlink has no __module__ attribute"
)

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import logging

# 配置日志输出
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

from clients.qwen_client import QwenClient
from config import QwenVLConfig
from tests.test_data import get_test_image_base64, get_test_landscape


def test_with_real_api():
    """使用真实API测试并显示日志"""
    print("🧪 测试真实API调用和日志功能")
    print("=" * 60)

    # 检查环境变量
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("❌ 未设置DASHSCOPE_API_KEY环境变量")
        return False

    use_real_api = os.getenv("USE_REAL_API", "false").lower()
    if use_real_api != "true":
        print("❌ 未设置USE_REAL_API=true")
        return False

    print(f"✅ API密钥: {api_key[:10]}...")
    print("✅ 真实API已启用")

    # 创建配置
    config = QwenVLConfig(api_key=api_key)

    # 创建启用日志的客户端
    client = QwenClient(qwen_config=config, enable_logging=True)

    try:
        print("\n🔍 测试1: 文本聊天")
        print("-" * 40)

        result1 = client.chat_with_text(
            user_prompt="你好，请用一句话简单介绍你自己",
            system_prompt="你是一个简洁的AI助手",
            temperature=0.5,
            max_tokens=100,
        )

        print(f"\n📋 文本聊天结果: {result1}")

        print("\n🔍 测试2: 图片分析")
        print("-" * 40)

        # 获取真实图片数据
        image_path = get_test_landscape()
        image_base64 = get_test_image_base64(image_path)

        print(f"📸 使用图片: {image_path}")
        print(f"📊 图片数据大小: {len(image_base64)} 字符")

        result2 = client.chat_with_image(
            image_base64=image_base64,
            user_prompt="请用一句话简单描述这张图片的主要内容",
            system_prompt="你是一个图像分析助手，请简洁地描述图片内容",
            temperature=0.7,
            max_tokens=200,
        )

        print(f"\n📋 图片分析结果: {result2}")

        print("\n" + "=" * 60)
        print("🎉 所有测试完成！")
        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_without_logging():
    """测试禁用日志的情况"""
    print("\n🔇 测试禁用日志功能")
    print("-" * 40)

    api_key = os.getenv("DASHSCOPE_API_KEY")
    config = QwenVLConfig(api_key=api_key)

    # 创建禁用日志的客户端
    client = QwenClient(qwen_config=config, enable_logging=False)

    try:
        result = client.chat_with_text(
            user_prompt="这个测试不会显示API调用日志", temperature=0.3, max_tokens=50
        )

        print(f"📋 禁用日志测试结果: {result}")
        return True

    except Exception as e:
        print(f"❌ 禁用日志测试失败: {e}")
        return False


if __name__ == "__main__":
    success1 = test_with_real_api()
    success2 = test_without_logging()

    print(f"\n📊 测试总结:")
    print(f"   - 启用日志测试: {'✅ 成功' if success1 else '❌ 失败'}")
    print(f"   - 禁用日志测试: {'✅ 成功' if success2 else '❌ 失败'}")

    sys.exit(0 if success1 and success2 else 1)
