#!/usr/bin/env python3
"""
QwenRag 演示脚本

展示图片处理和FAISS存储的基本功能
"""

import os
import sys
import tempfile
from pathlib import Path
from PIL import Image, ImageDraw

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import QwenVLConfig, ImageProcessorConfig
from processors.image_processor import ImageProcessor
from vector_store.faiss_store import FaissStore
from utils.logger import setup_logger
from schemas.data_models import ProcessingStatus
import numpy as np


def create_demo_images(temp_dir: str) -> list:
    """创建演示图片"""
    print("创建演示图片...")
    
    image_paths = []
    
    # 创建风景照
    landscape = Image.new('RGB', (400, 300), color='skyblue')
    draw = ImageDraw.Draw(landscape)
    draw.polygon([(50, 200), (200, 100), (350, 200)], fill='green')  # 山
    draw.ellipse([150, 80, 250, 120], fill='yellow')  # 太阳
    landscape_path = os.path.join(temp_dir, "landscape.jpg")
    landscape.save(landscape_path)
    image_paths.append(landscape_path)
    
    # 创建人物照
    portrait = Image.new('RGB', (300, 400), color='lightgray')
    draw = ImageDraw.Draw(portrait)
    draw.ellipse([100, 150, 200, 250], fill='peachpuff')  # 脸
    draw.ellipse([120, 180, 135, 195], fill='black')  # 左眼
    draw.ellipse([165, 180, 180, 195], fill='black')  # 右眼
    draw.arc([130, 210, 170, 230], 0, 180, fill='red', width=3)  # 嘴
    portrait_path = os.path.join(temp_dir, "portrait.jpg")
    portrait.save(portrait_path)
    image_paths.append(portrait_path)
    
    # 创建截图
    screenshot = Image.new('RGB', (300, 500), color='white')
    draw = ImageDraw.Draw(screenshot)
    draw.rectangle([0, 0, 300, 50], fill='blue')  # 标题栏
    draw.text([10, 20], "App Demo", fill='white')
    draw.rectangle([20, 80, 280, 120], fill='lightblue')  # 按钮
    draw.text([30, 95], "Button 1", fill='black')
    screenshot_path = os.path.join(temp_dir, "screenshot.png")
    screenshot.save(screenshot_path)
    image_paths.append(screenshot_path)
    
    print(f"创建了 {len(image_paths)} 张演示图片")
    return image_paths


def demo_image_processing(image_paths: list, use_real_api: bool = False):
    """演示图片处理功能"""
    print("\n=== 图片处理演示 ===")
    
    if use_real_api:
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            print("跳过真实API演示 - 需要设置DASHSCOPE_API_KEY环境变量")
            return []
        
        print("使用真实Qwen VL API...")
        config = QwenVLConfig(api_key=api_key)
        processor = ImageProcessor(qwen_config=config)
        
        results = []
        for i, image_path in enumerate(image_paths):
            print(f"处理图片 {i+1}/{len(image_paths)}: {Path(image_path).name}")
            result = processor.process_image(image_path)
            
            if result.processing_status == ProcessingStatus.SUCCESS:
                print(f"  ✓ 成功: {result.description}")
                print(f"  标签: 风景={result.is_landscape}, 人物={result.has_person}, 截图={result.is_snap}")
            else:
                print(f"  ✗ 失败: {result.error_message}")
            
            results.append(result)
        
        return results
    
    else:
        print("使用模拟数据演示...")
        from schemas.data_models import ImageMetadata
        
        # 创建模拟结果
        results = []
        descriptions = [
            "美丽的山水风景，蓝天白云，绿色的山峰",
            "一个人的肖像照片，面部特征清晰", 
            "手机应用界面截图，包含标题栏和按钮"
        ]
        
        for i, (image_path, desc) in enumerate(zip(image_paths, descriptions)):
            result = ImageMetadata(
                path=image_path,
                is_landscape=(i == 0),
                has_person=(i == 1), 
                is_snap=(i == 2),
                description=desc,
                unique_id=f"demo_id_{i}",
                processing_status=ProcessingStatus.SUCCESS
            )
            results.append(result)
            
            print(f"模拟处理: {Path(image_path).name}")
            print(f"  描述: {desc}")
            print(f"  标签: 风景={result.is_landscape}, 人物={result.has_person}, 截图={result.is_snap}")
        
        return results


def demo_faiss_storage(metadata_list: list):
    """演示FAISS存储功能"""
    print("\n=== FAISS存储演示 ===")
    
    # 创建FAISS存储
    print("初始化FAISS存储...")
    faiss_store = FaissStore(dimension=768)
    
    # 生成模拟向量
    print("生成模拟向量...")
    vectors = []
    ids = []
    
    for meta in metadata_list:
        if meta.processing_status == ProcessingStatus.SUCCESS:
            # 生成基于描述的模拟向量
            vector = np.random.rand(768).astype(np.float32)
            # 添加一些基于内容的特征
            if meta.is_landscape:
                vector[0:100] += 0.5  # 风景特征
            if meta.has_person:
                vector[100:200] += 0.5  # 人物特征
            if meta.is_snap:
                vector[200:300] += 0.5  # 截图特征
                
            vectors.append(vector)
            ids.append(meta.unique_id)
    
    if not vectors:
        print("没有成功处理的图片，跳过向量存储演示")
        return
    
    vectors_array = np.array(vectors)
    
    # 添加向量
    print(f"添加 {len(vectors)} 个向量到FAISS索引...")
    success = faiss_store.add_vectors(vectors_array, ids)
    print(f"添加向量: {'成功' if success else '失败'}")
    
    # 搜索演示
    print("\n搜索演示:")
    query_vector = vectors_array[0:1]  # 使用第一个向量作为查询
    distances, indices, result_ids = faiss_store.search(query_vector, k=3)
    
    print(f"查询向量: {ids[0]}")
    print("搜索结果:")
    for i, (dist, idx, result_id) in enumerate(zip(distances, indices, result_ids)):
        print(f"  {i+1}. ID: {result_id}, 距离: {dist:.4f}")
    
    # 获取统计信息
    stats = faiss_store.get_statistics()
    print(f"\n索引统计:")
    print(f"  总向量数: {stats['total_vectors']}")
    print(f"  向量维度: {stats['dimension']}")
    print(f"  索引类型: {stats['index_type']}")
    
    return faiss_store


def main():
    """主演示函数"""
    print("🚀 QwenRag 系统演示")
    print("=" * 50)
    
    # 设置日志
    logger = setup_logger(log_level="INFO")
    
    # 检查是否使用真实API
    use_real_api = os.getenv("USE_REAL_API", "false").lower() == "true"
    
    if use_real_api:
        print("📡 将使用真实的Qwen VL API")
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            print("⚠️  警告: 未设置DASHSCOPE_API_KEY，将使用模拟数据")
            use_real_api = False
    else:
        print("🎭 使用模拟数据演示")
    
    try:
        # 创建临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"临时目录: {temp_dir}")
            
            # 创建演示图片
            image_paths = create_demo_images(temp_dir)
            
            # 演示图片处理
            metadata_list = demo_image_processing(image_paths, use_real_api)
            
            # 演示FAISS存储
            faiss_store = demo_faiss_storage(metadata_list)
            
            print("\n✅ 演示完成！")
            print("\n📝 总结:")
            print(f"  - 创建了 {len(image_paths)} 张测试图片")
            print(f"  - 处理了 {len(metadata_list)} 张图片")
            success_count = sum(1 for m in metadata_list if m.processing_status == ProcessingStatus.SUCCESS)
            print(f"  - 成功处理 {success_count} 张图片")
            
            if faiss_store:
                stats = faiss_store.get_statistics()
                print(f"  - FAISS索引包含 {stats['total_vectors']} 个向量")
            
            print("\n🎯 下一步:")
            print("  1. 设置DASHSCOPE_API_KEY环境变量")
            print("  2. 运行 python main_index.py 索引真实图片")
            print("  3. 运行 python main_search.py 搜索图片")
            print("  4. 运行 pytest tests/ 执行测试")
    
    except KeyboardInterrupt:
        print("\n用户中断演示")
    except Exception as e:
        print(f"\n❌ 演示过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()