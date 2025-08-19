"""
端到端完整流程演示
演示从图片解析 -> 向量化存储 -> 检索查询的完整过程
"""
import os
import sys
import logging
from pathlib import Path

# 添加项目根目录到路径
sys.path.append('.')

from pipelines.indexing_pipeline import IndexingPipeline
from pipelines.retrieval_pipeline import RetrievalPipeline
from utils.logger import logger

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """主函数 - 演示完整流程"""
    print("🚀 QwenRag 端到端流程演示")
    print("=" * 50)
    
    # 配置路径
    dataset_dir = "dataset"
    metadata_path = "demo_metadata.json"
    index_path = "demo_index.faiss"
    
    # 检查数据集目录
    if not Path(dataset_dir).exists():
        print(f"❌ 数据集目录不存在: {dataset_dir}")
        return
    
    try:
        # Step 1: 构建索引
        print("\n📚 Step 1: 构建索引流水线")
        print("-" * 30)
        
        indexing_pipeline = IndexingPipeline(
            metadata_save_path=metadata_path,
            batch_size=5,
            max_workers=2,
            auto_save=True
        )
        
        # 构建索引
        indexing_results = indexing_pipeline.build_index_from_directory(
            directory_path=dataset_dir,
            recursive=True,
            parallel=True,
            resume_from_metadata=True
        )
        
        print(f"✅ 索引构建完成!")
        print(f"   总计: {indexing_results['total']} 个文件")
        print(f"   成功: {indexing_results['success']} 个")
        print(f"   失败: {indexing_results['failed']} 个")
        
        if indexing_results['success'] == 0:
            print("❌ 没有成功处理的文件，退出演示")
            return
        
        # Step 2: 检索演示
        print("\n🔍 Step 2: 检索流水线演示")
        print("-" * 30)
        
        retrieval_pipeline = RetrievalPipeline(
            metadata_path=metadata_path,
            default_top_k=5,
            similarity_threshold=0.3
        )
        
        # 演示不同的查询类型
        demo_queries = [
            "游戏画面",
            "角色",
            "风景",
            "截图"
        ]
        
        print("\n🎯 文本查询演示:")
        for query in demo_queries:
            print(f"\n查询: '{query}'")
            results = retrieval_pipeline.search_by_text(query, top_k=3)
            
            if results:
                for i, result in enumerate(results[:3], 1):
                    print(f"  {i}. {Path(result['image_path']).name}")
                    print(f"     相似度: {result['similarity_score']:.3f}")
                    print(f"     匹配类型: {result['match_type']}")
                    if result.get('metadata', {}).get('description'):
                        desc = result['metadata']['description'][:50] + "..."
                        print(f"     描述: {desc}")
            else:
                print("  未找到匹配结果")
        
        # 演示图片查询
        print("\n🖼️  以图搜图演示:")
        
        # 找一张图片作为查询样本
        sample_images = list(Path(dataset_dir).glob("*.jpg"))[:3]
        
        for sample_image in sample_images:
            print(f"\n查询图片: {sample_image.name}")
            results = retrieval_pipeline.search_by_image(str(sample_image), top_k=3)
            
            if results:
                for i, result in enumerate(results[:3], 1):
                    result_name = Path(result['image_path']).name
                    if result_name != sample_image.name:  # 排除自己
                        print(f"  {i}. {result_name}")
                        print(f"     相似度: {result['similarity_score']:.3f}")
                        print(f"     匹配类型: {result['match_type']}")
            else:
                print("  未找到相似图片")
        
        # Step 3: 统计信息
        print("\n📊 Step 3: 系统统计")
        print("-" * 30)
        
        indexing_stats = indexing_pipeline.get_statistics()
        retrieval_stats = retrieval_pipeline.get_statistics()
        
        print(f"📚 索引统计:")
        print(f"   处理总数: {indexing_stats['total_processed']}")
        print(f"   成功数量: {indexing_stats['success_count']}")
        print(f"   失败数量: {indexing_stats['failed_count']}")
        print(f"   元数据数: {indexing_stats['metadata_count']}")
        
        print(f"\n🔍 检索统计:")
        print(f"   向量数量: {retrieval_stats['vector_count']}")
        print(f"   元数据数: {retrieval_stats['metadata_count']}")
        print(f"   向量维度: {retrieval_stats['index_dimension']}")
        print(f"   相似度阈值: {retrieval_stats['similarity_threshold']}")
        
        print("\n🎉 端到端流程演示完成!")
        print("=" * 50)
        
    except Exception as e:
        logger.error(f"演示过程出错: {e}", exc_info=True)
        print(f"❌ 演示失败: {e}")


if __name__ == "__main__":
    # 检查API密钥
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("⚠️  警告: 未设置 DASHSCOPE_API_KEY 环境变量")
        print("   某些功能可能无法正常工作")
        print("")
    
    main()