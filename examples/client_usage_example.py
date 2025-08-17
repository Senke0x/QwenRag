#!/usr/bin/env python3
"""
QwenClient 和 PromptManager 使用示例

这个示例展示了如何使用重构后的统一客户端和提示词管理器
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import QwenVLConfig
from clients.qwen_client import QwenClient
from clients.prompt_manager import PromptManager, PromptType
from processors.image_processor import ImageProcessor


def example_qwen_client_basic():
    """基础 QwenClient 使用示例"""
    print("=== QwenClient 基础使用示例 ===")
    
    # 初始化客户端
    config = QwenVLConfig(api_key=os.getenv("DASHSCOPE_API_KEY", "your_api_key_here"))
    client = QwenClient(qwen_config=config)
    
    print(f"客户端信息: {client.get_client_info()}")
    
    # 如果有真实的 API key，可以测试文本聊天
    if config.api_key and config.api_key != "your_api_key_here":
        try:
            response = client.chat_with_text(
                user_prompt="请简单介绍一下人工智能的发展历史",
                system_prompt="你是一个知识渊博的AI助手"
            )
            print(f"文本聊天响应: {response[:100]}...")
        except Exception as e:
            print(f"API 调用失败: {e}")
    else:
        print("未提供有效的 API key，跳过实际 API 调用")


def example_prompt_manager():
    """PromptManager 使用示例"""
    print("\n=== PromptManager 使用示例 ===")
    
    # 初始化提示词管理器
    pm = PromptManager()
    
    # 列出所有可用的提示词类型
    print("可用的提示词类型:")
    for prompt_type in pm.list_prompt_types():
        print(f"  - {prompt_type}")
    
    # 获取图像分析提示词
    image_prompt = pm.get_prompt(PromptType.IMAGE_ANALYSIS)
    print(f"\n图像分析提示词:")
    print(f"系统提示词: {image_prompt['system']}")
    print(f"用户提示词: {image_prompt['user'][:100]}...")
    
    # 获取人脸检测提示词
    face_prompt = pm.get_prompt(PromptType.FACE_DETECTION)
    print(f"\n人脸检测用户提示词: {face_prompt['user'][:100]}...")
    
    # 添加自定义提示词
    pm.add_prompt(
        prompt_type="custom_test",
        system_prompt="你是一个测试助手",
        user_prompt="请帮我测试这个功能：{test_content}"
    )
    
    # 使用自定义提示词（通过字符串访问）
    custom_prompts = pm._prompts["custom_test"]
    custom_prompt_formatted = {
        "system": custom_prompts["system"],
        "user": custom_prompts["user"].format(test_content="图像识别")
    }
    print(f"\n自定义提示词: {custom_prompt_formatted['user']}")


def example_integrated_usage():
    """集成使用示例"""
    print("\n=== 集成使用示例 ===")
    
    # 创建统一的组件
    config = QwenVLConfig(api_key=os.getenv("DASHSCOPE_API_KEY", "test_key"))
    qwen_client = QwenClient(qwen_config=config)
    prompt_manager = PromptManager()
    
    # 创建图片处理器
    processor = ImageProcessor(
        qwen_client=qwen_client,
        prompt_manager=prompt_manager
    )
    
    print("✓ 所有组件初始化成功")
    print(f"  - QwenClient: {qwen_client.qwen_config.model}")
    print(f"  - PromptManager: {len(prompt_manager.list_prompt_types())} 种提示词类型")
    print(f"  - ImageProcessor: 已准备就绪")
    
    # 展示客户端复用
    print("\n客户端复用示例:")
    print("✓ 同一个 QwenClient 实例可以被多个组件共享")
    print("✓ PromptManager 可以统一管理所有提示词")
    print("✓ 减少了代码重复和配置管理")


def example_advanced_prompt_usage():
    """高级提示词使用示例"""
    print("\n=== 高级提示词使用示例 ===")
    
    pm = PromptManager()
    
    # 更新现有提示词
    pm.update_prompt(
        PromptType.SCENE_CLASSIFICATION,
        user_prompt="请识别这张图片的场景类型，重点关注：{focus_areas}"
    )
    
    # 使用参数化提示词
    scene_prompt = pm.get_prompt(
        PromptType.SCENE_CLASSIFICATION,
        focus_areas="室内外环境、光照条件、主要物体"
    )
    
    print("参数化提示词示例:")
    print(f"用户提示词: {scene_prompt['user']}")
    
    # 展示所有提示词
    print("\n所有提示词概览:")
    all_prompts = pm.get_all_prompts()
    for prompt_type, prompts in all_prompts.items():
        print(f"  {prompt_type}:")
        print(f"    系统: {prompts['system'][:50]}...")
        print(f"    用户: {prompts['user'][:50]}...")


def main():
    """主函数"""
    print("🚀 QwenRag 重构后的客户端使用示例")
    print("=" * 60)
    
    try:
        example_qwen_client_basic()
        example_prompt_manager()
        example_integrated_usage()
        example_advanced_prompt_usage()
        
        print("\n" + "=" * 60)
        print("✅ 所有示例运行完成！")
        print("\n💡 重构的优势:")
        print("  1. 统一的 API 客户端，减少重复代码")
        print("  2. 集中的提示词管理，便于维护和扩展")
        print("  3. 更好的错误处理和重试机制")
        print("  4. 组件间的松耦合，提高可测试性")
        print("  5. 支持多种提示词类型和参数化")
        
    except Exception as e:
        print(f"❌ 示例运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()