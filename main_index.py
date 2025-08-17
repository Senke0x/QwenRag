#!/usr/bin/env python3
"""
图片索引主程序

使用方法:
    python main_index.py --input_dir /path/to/images --output_dir /path/to/index

示例:
    python main_index.py --input_dir ./test_images --output_dir ./index --api_key your_api_key
"""

import argparse
import os
import sys
import logging
from typing import List
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import QwenVLConfig, ImageProcessorConfig
from processors.image_processor import ImageProcessor
from clients.qwen_client import QwenClient
from vector_store.faiss_store import FaissStore
from utils.logger import setup_logger
from utils.image_utils import find_images_in_directory
from schemas.data_models import ProcessingStatus


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """设置日志记录"""
    return setup_logger(
        name="qwen_rag_indexing",
        log_file=log_file,
        log_level=log_level
    )


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="QwenRag图片索引工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--input_dir", "-i",
        required=True,
        help="输入图片目录路径"
    )
    
    parser.add_argument(
        "--output_dir", "-o", 
        required=True,
        help="输出索引目录路径"
    )
    
    parser.add_argument(
        "--api_key", "-k",
        help="Qwen API密钥 (也可通过DASHSCOPE_API_KEY环境变量设置)"
    )
    
    parser.add_argument(
        "--batch_size", "-b",
        type=int,
        default=10,
        help="批处理大小 (默认: 10)"
    )
    
    parser.add_argument(
        "--log_level", "-l",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="日志级别 (默认: INFO)"
    )
    
    parser.add_argument(
        "--log_file",
        help="日志文件路径 (默认: 只输出到控制台)"
    )
    
    parser.add_argument(
        "--resume", "-r",
        action="store_true",
        help="恢复之前中断的索引任务"
    )
    
    parser.add_argument(
        "--skip_existing",
        action="store_true", 
        help="跳过已经处理过的图片"
    )
    
    return parser.parse_args()


def create_output_directory(output_dir: str) -> bool:
    """创建输出目录"""
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logging.error(f"创建输出目录失败: {e}")
        return False


def save_metadata(metadata_list: List, output_dir: str):
    """保存元数据到JSON文件"""
    import json
    
    metadata_file = Path(output_dir) / "metadata.json"
    
    try:
        # 转换为可序列化的格式
        serializable_data = [meta.to_dict() for meta in metadata_list]
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, ensure_ascii=False, indent=2)
        
        logging.info(f"元数据已保存到: {metadata_file}")
        
    except Exception as e:
        logging.error(f"保存元数据失败: {e}")


def main():
    """主函数"""
    args = parse_arguments()
    
    # 设置日志
    logger = setup_logging(args.log_level, args.log_file)
    
    logger.info("=== QwenRag 图片索引工具启动 ===")
    logger.info(f"输入目录: {args.input_dir}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"批处理大小: {args.batch_size}")
    
    # 验证输入目录
    if not Path(args.input_dir).exists():
        logger.error(f"输入目录不存在: {args.input_dir}")
        sys.exit(1)
    
    # 创建输出目录
    if not create_output_directory(args.output_dir):
        sys.exit(1)
    
    # 配置API
    api_key = args.api_key or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        logger.error("请提供API密钥 (--api_key 或 DASHSCOPE_API_KEY 环境变量)")
        sys.exit(1)
    
    try:
        # 初始化配置
        qwen_config = QwenVLConfig(api_key=api_key)
        image_config = ImageProcessorConfig()
        
        # 初始化处理器
        logger.info("初始化图片处理器...")
        qwen_client = QwenClient(qwen_config=qwen_config)
        processor = ImageProcessor(qwen_client=qwen_client)
        
        # 查找图片文件
        logger.info("扫描图片文件...")
        image_files = find_images_in_directory(args.input_dir, image_config)
        
        if not image_files:
            logger.warning(f"在目录 {args.input_dir} 中未找到支持的图片文件")
            sys.exit(0)
        
        logger.info(f"找到 {len(image_files)} 个图片文件")
        
        # 批量处理图片
        all_metadata = []
        failed_count = 0
        
        for i in range(0, len(image_files), args.batch_size):
            batch = image_files[i:i + args.batch_size]
            batch_num = i // args.batch_size + 1
            total_batches = (len(image_files) + args.batch_size - 1) // args.batch_size
            
            logger.info(f"处理批次 {batch_num}/{total_batches} ({len(batch)} 张图片)")
            
            # 处理当前批次
            batch_results = processor.process_images_batch(batch)
            all_metadata.extend(batch_results)
            
            # 统计结果
            success_count = sum(1 for r in batch_results if r.processing_status == ProcessingStatus.SUCCESS)
            batch_failed = len(batch_results) - success_count
            failed_count += batch_failed
            
            logger.info(f"批次 {batch_num} 完成: 成功 {success_count}, 失败 {batch_failed}")
        
        # 保存元数据
        logger.info("保存处理结果...")
        save_metadata(all_metadata, args.output_dir)
        
        # 创建FAISS索引 (如果有成功的处理结果)
        successful_metadata = [m for m in all_metadata if m.processing_status == ProcessingStatus.SUCCESS]
        
        if successful_metadata:
            logger.info(f"为 {len(successful_metadata)} 张图片创建向量索引...")
            
            # 注意: 这里需要实际的embedding功能，目前只创建空索引
            faiss_store = FaissStore(dimension=768)
            
            # 保存索引结构
            index_file = Path(args.output_dir) / "semantic_index.faiss"
            faiss_store.save_index(str(index_file))
            
            logger.info(f"向量索引已保存到: {index_file}")
        
        # 输出最终统计
        total_processed = len(all_metadata)
        success_count = len(successful_metadata)
        
        logger.info("=== 索引完成 ===")
        logger.info(f"总处理图片: {total_processed}")
        logger.info(f"成功处理: {success_count}")
        logger.info(f"处理失败: {failed_count}")
        logger.info(f"成功率: {success_count/total_processed*100:.1f}%")
        
        if failed_count > 0:
            logger.warning(f"有 {failed_count} 张图片处理失败，请检查日志获取详细信息")
    
    except KeyboardInterrupt:
        logger.info("用户中断操作")
        sys.exit(1)
    except Exception as e:
        logger.error(f"程序执行出错: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()