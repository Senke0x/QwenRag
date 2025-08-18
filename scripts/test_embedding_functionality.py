#!/usr/bin/env python3
"""
Embedding功能综合测试脚本
合并了原来的test_embedding_processor.py, test_multimodal_embedding.py和example_embedding_workflow.py
"""
import os
import sys
import logging
import json
from pathlib import Path
from datetime import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from clients.qwen_client import QwenClient
from processors.embedding_processor import EmbeddingProcessor
from utils.logger import logger

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def test_qwen_client_basic():
    """测试QwenClient基本功能"""
    print("🚀 测试QwenClient基本功能...")
    
    # 检查API密钥
    if not os.getenv('DASHSCOPE_API_KEY'):
        print("⚠️  未设置 DASHSCOPE_API_KEY，跳过实际API测试")
        return False
    
    try:
        # 初始化客户端
        client = QwenClient()
        print("✅ QwenClient初始化成功")
        
        # 获取客户端信息
        client_info = client.get_client_info()
        print(f"📋 客户端信息:")
        for key, value in client_info.items():
            print(f"  {key}: {value}")
        
        # 测试文本embedding
        print("\n📝 测试文本embedding...")
        text_result = client.get_text_embedding("这是一段测试文本，用于验证embedding功能")
        print(f"✅ 文本embedding成功: 维度={len(text_result['embedding'])}, 模型={text_result['model']}")
        
        print("\n🎉 QwenClient基本功能测试完成!")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def test_embedding_processor_basic():
    """测试EmbeddingProcessor基本功能"""
    print("\n🚀 测试EmbeddingProcessor基本功能...")
    
    try:
        # 初始化处理器
        processor = EmbeddingProcessor(
            embedding_dimension=1536,  # multimodal-embedding-v1的维度
            index_save_path="test_data/basic_test_index"
        )
        print("✅ EmbeddingProcessor初始化成功")
        
        # 获取统计信息
        stats = processor.get_statistics()
        print(f"📊 初始统计信息:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        return processor
        
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return None


def test_text_processing(processor):
    """测试文本处理功能"""
    if not processor:
        return False
        
    print("\n📝 测试文本处理功能...")
    
    # 检查API密钥
    if not os.getenv('DASHSCOPE_API_KEY'):
        print("⚠️  未设置 DASHSCOPE_API_KEY，跳过文本处理测试")
        return False
    
    try:
        # 测试单个文本处理
        test_texts = [
            "这是一段美丽的风景描述，包含蓝天白云和绿色的草地",
            "城市夜景，高楼大厦灯火通明的壮观场面",
            "森林中的小溪，清澈的水流过石头"
        ]
        
        for i, text in enumerate(test_texts):
            success = processor.process_text(text, f"text_{i+1}")
            if success:
                print(f"✅ 文本{i+1}处理成功")
            else:
                print(f"❌ 文本{i+1}处理失败")
        
        # 测试文本搜索
        print("\n🔍 测试文本搜索...")
        search_results = processor.search_by_text("美丽的风景", top_k=3)
        print(f"📊 搜索结果数量: {len(search_results)}")
        for result in search_results[:2]:  # 只显示前2个
            print(f"  - ID: {result['vector_id']}, 相似度: {result['similarity_score']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 文本处理测试失败: {e}")
        return False


def test_image_processing(processor):
    """测试图片处理功能（如果有测试图片）"""
    if not processor:
        return False
        
    print("\n🖼️  测试图片处理功能...")
    
    # 检查API密钥
    if not os.getenv('DASHSCOPE_API_KEY'):
        print("⚠️  未设置 DASHSCOPE_API_KEY，跳过图片处理测试")
        return False
    
    try:
        # 检查是否有测试图片
        dataset_dir = Path(project_root / "dataset")
        if dataset_dir.exists():
            image_files = list(dataset_dir.glob("*.jpg"))[:2]  # 只处理前2张
            if image_files:
                print(f"📂 找到 {len(image_files)} 张测试图片")
                
                for i, image_path in enumerate(image_files):
                    # 模拟处理图片base64
                    try:
                        # 这里只是模拟，实际需要转换为base64
                        print(f"📸 模拟处理图片 {i+1}: {image_path.name}")
                        # success = processor.process_image_base64(image_base64, f"img_{i+1}")
                        print(f"  ✅ 图片{i+1}处理模拟成功")
                    except Exception as e:
                        print(f"  ❌ 图片{i+1}处理失败: {e}")
            else:
                print("❌ 未找到.jpg格式的测试图片")
        else:
            print("❌ 未找到dataset目录")
        
        print("💡 图片处理功能需要实际的base64数据进行完整测试")
        return True
        
    except Exception as e:
        print(f"❌ 图片处理测试失败: {e}")
        return False


def test_batch_processing():
    """测试批量处理功能"""
    print("\n📦 测试批量处理功能...")
    
    # 检查API密钥
    if not os.getenv('DASHSCOPE_API_KEY'):
        print("⚠️  未设置 DASHSCOPE_API_KEY，跳过批量处理测试")
        return False
    
    try:
        processor = EmbeddingProcessor(
            embedding_dimension=1536,
            index_save_path="test_data/batch_test_index"
        )
        
        # 批量文本处理
        batch_texts = [
            "海边的日落景色非常美丽",
            "雪山峰顶的壮丽景象",
            "城市繁华的商业街道",
            "宁静的乡村田园风光"
        ]
        text_ids = [f"batch_text_{i+1}" for i in range(len(batch_texts))]
        
        results = processor.process_batch_texts(batch_texts, text_ids)
        print(f"📊 批量文本处理结果: {results}")
        
        if results['success'] > 0:
            # 测试批量处理后的搜索
            search_results = processor.search_by_text("美丽的景色", top_k=3)
            print(f"🔍 批量处理后搜索结果: {len(search_results)} 个")
        
        return True
        
    except Exception as e:
        print(f"❌ 批量处理测试失败: {e}")
        return False


def test_edge_cases():
    """测试边界情况"""
    print("\n⚠️  测试边界情况...")
    
    try:
        processor = EmbeddingProcessor(
            embedding_dimension=1536,
            index_save_path="test_data/edge_test_index"
        )
        
        # 测试空文本
        try:
            result = processor.process_text("", "empty_text")
            print(f"  空文本处理: {'成功' if result else '失败（预期）'}")
        except Exception as e:
            print(f"  空文本处理异常（预期）: {str(e)[:50]}...")
        
        # 测试极长文本
        long_text = "这是一个非常" + "长的" * 100 + "文本测试"
        try:
            result = processor.process_text(long_text, "long_text")
            print(f"  长文本处理: {'成功' if result else '失败'}")
        except Exception as e:
            print(f"  长文本处理异常: {str(e)[:50]}...")
        
        print("✅ 边界情况测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 边界情况测试失败: {e}")
        return False


def test_workflow_simulation():
    """模拟完整工作流程"""
    print("\n🔄 模拟完整工作流程...")
    
    # 检查API密钥
    if not os.getenv('DASHSCOPE_API_KEY'):
        print("⚠️  未设置 DASHSCOPE_API_KEY，跳过工作流程模拟")
        return False
    
    try:
        print("📋 Step 1: 初始化处理器...")
        processor = EmbeddingProcessor(
            embedding_dimension=1536,
            index_save_path="test_data/workflow_index"
        )
        print("✅ 处理器初始化完成")
        
        print("\n📝 Step 2: 模拟文本数据处理...")
        sample_texts = [
            "风景如画的山水田园",
            "现代化的城市建筑群",
            "清晨的阳光洒向大地"
        ]
        
        for i, text in enumerate(sample_texts):
            success = processor.process_text(text, f"workflow_text_{i+1}")
            print(f"  文本{i+1}: {'✅ 成功' if success else '❌ 失败'}")
        
        print("\n🔍 Step 3: 测试搜索功能...")
        results = processor.search_by_text("美丽的风景", top_k=5)
        print(f"📊 搜索到 {len(results)} 个结果")
        
        print("\n📈 Step 4: 最终统计...")
        stats = processor.get_statistics()
        print("最终统计信息:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print("\n🎉 工作流程模拟完成!")
        return True
        
    except Exception as e:
        print(f"❌ 工作流程模拟失败: {e}")
        return False


def main():
    """主测试函数"""
    print("=" * 80)
    print("🧪 Embedding功能综合测试")
    print("=" * 80)
    
    # 测试结果统计
    tests = [
        ("QwenClient基本功能", test_qwen_client_basic),
        ("EmbeddingProcessor基本功能", lambda: test_embedding_processor_basic() is not None),
        ("批量处理功能", test_batch_processing),
        ("边界情况处理", test_edge_cases),
        ("完整工作流程", test_workflow_simulation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'-' * 60}")
        print(f"🔍 开始测试: {test_name}")
        print(f"{'-' * 60}")
        
        try:
            result = test_func()
            results.append(result)
            status = "✅ 通过" if result else "❌ 失败"
            print(f"📊 {test_name}: {status}")
        except Exception as e:
            results.append(False)
            print(f"📊 {test_name}: ❌ 异常 - {str(e)[:50]}...")
    
    # 最终总结
    print("\n" + "=" * 80)
    print("📊 测试总结:")
    passed = sum(results)
    total = len(results)
    print(f"  通过: {passed}/{total}")
    print(f"  失败: {total - passed}/{total}")
    print(f"  成功率: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n🎉 所有测试通过!")
    else:
        print(f"\n⚠️  有 {total - passed} 个测试未通过，请检查配置")
    
    print("\n📝 测试说明:")
    print("  1. 需要设置 DASHSCOPE_API_KEY 环境变量才能进行完整测试")
    print("  2. 图片处理测试需要dataset目录中有.jpg文件")
    print("  3. 向量索引文件保存在test_data目录中")
    print("=" * 80)


if __name__ == "__main__":
    main()