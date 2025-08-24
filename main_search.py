#!/usr/bin/env python3
"""
图片搜索主程序

使用方法:
    python main_search.py --index_dir /path/to/index --query "山水风景"
    python main_search.py --index_dir /path/to/index --image_query /path/to/query.jpg

示例:
    python main_search.py --index_dir ./index --query "美丽的风景" --top_k 5
    python main_search.py --index_dir ./index --image_query ./query.jpg --top_k 3
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from clients.qwen_client import QwenClient
from config import QwenVLConfig
from processors.image_processor import ImageProcessor
from schemas.data_models import ImageMetadata, ProcessingStatus, SearchResult
from utils.logger import setup_logger
from vector_store.faiss_store import FaissStore


def setup_logging(log_level: str = "INFO"):
    """设置日志记录"""
    return setup_logger(name="qwen_rag_search", log_level=log_level)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="QwenRag图片搜索工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--index_dir", "-i", required=True, help="索引目录路径")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--query", "-q", help="文本查询内容")
    group.add_argument("--image_query", "-img", help="查询图片路径")

    parser.add_argument("--top_k", "-k", type=int, default=5, help="返回结果数量 (默认: 5)")

    parser.add_argument("--api_key", help="Qwen API密钥 (用于图片查询)")

    parser.add_argument(
        "--log_level",
        "-l",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="日志级别 (默认: INFO)",
    )

    parser.add_argument(
        "--output_format",
        "-f",
        choices=["json", "table", "simple"],
        default="table",
        help="输出格式 (默认: table)",
    )

    parser.add_argument("--show_similarity", action="store_true", help="显示相似度分数")

    return parser.parse_args()


def load_metadata(index_dir: str) -> List[ImageMetadata]:
    """加载元数据"""
    metadata_file = Path(index_dir) / "metadata.json"

    if not metadata_file.exists():
        raise FileNotFoundError(f"元数据文件不存在: {metadata_file}")

    try:
        with open(metadata_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        return [ImageMetadata.from_dict(item) for item in data]

    except Exception as e:
        raise Exception(f"加载元数据失败: {e}")


def load_faiss_index(index_dir: str) -> FaissStore:
    """加载FAISS索引"""
    index_file = Path(index_dir) / "semantic_index.faiss"

    if not index_file.exists():
        raise FileNotFoundError(f"FAISS索引文件不存在: {index_file}")

    try:
        faiss_store = FaissStore(dimension=768)
        success = faiss_store.load_index(str(index_file))

        if not success:
            raise Exception("FAISS索引加载失败")

        return faiss_store

    except Exception as e:
        raise Exception(f"加载FAISS索引失败: {e}")


def text_search(
    query: str, metadata_list: List[ImageMetadata], top_k: int
) -> List[SearchResult]:
    """基于文本描述的搜索 (简单实现)"""
    results = []

    # 简单的文本匹配搜索
    for i, metadata in enumerate(metadata_list):
        if not metadata.description:
            continue

        # 计算简单的文本相似度 (关键词匹配)
        query_words = set(query.lower().split())
        desc_words = set(metadata.description.lower().split())

        # 计算交集比例作为相似度
        intersection = len(query_words & desc_words)
        union = len(query_words | desc_words)

        similarity = intersection / union if union > 0 else 0.0

        if similarity > 0:
            result = SearchResult(
                metadata=metadata,
                similarity_score=similarity,
                rank=i,
                search_type="text_simple",
            )
            results.append(result)

    # 按相似度排序
    results.sort(key=lambda x: x.similarity_score, reverse=True)

    return results[:top_k]


def image_search(
    image_path: str,
    metadata_list: List[ImageMetadata],
    processor: ImageProcessor,
    top_k: int,
) -> List[SearchResult]:
    """基于图片的搜索 (简单实现)"""
    try:
        # 处理查询图片
        query_metadata = processor.process_image(image_path)

        if query_metadata.processing_status != ProcessingStatus.SUCCESS:
            raise Exception(f"查询图片处理失败: {query_metadata.error_message}")

        # 简单的特征匹配
        results = []

        for i, metadata in enumerate(metadata_list):
            similarity = 0.0

            # 基于图片类型相似度
            if query_metadata.is_landscape == metadata.is_landscape:
                similarity += 0.3

            if query_metadata.has_person == metadata.has_person:
                similarity += 0.3

            if query_metadata.is_snap == metadata.is_snap:
                similarity += 0.2

            # 基于描述相似度
            if query_metadata.description and metadata.description:
                query_words = set(query_metadata.description.lower().split())
                desc_words = set(metadata.description.lower().split())

                intersection = len(query_words & desc_words)
                union = len(query_words | desc_words)

                text_sim = intersection / union if union > 0 else 0.0
                similarity += 0.2 * text_sim

            if similarity > 0:
                result = SearchResult(
                    metadata=metadata,
                    similarity_score=similarity,
                    rank=i,
                    search_type="image_simple",
                )
                results.append(result)

        # 按相似度排序
        results.sort(key=lambda x: x.similarity_score, reverse=True)

        return results[:top_k]

    except Exception as e:
        raise Exception(f"图片搜索失败: {e}")


def format_results(
    results: List[SearchResult], output_format: str, show_similarity: bool = False
):
    """格式化输出结果"""
    if not results:
        print("没有找到匹配的结果")
        return

    if output_format == "json":
        output_data = []
        for result in results:
            item = {
                "path": result.metadata.path,
                "description": result.metadata.description,
                "is_landscape": result.metadata.is_landscape,
                "has_person": result.metadata.has_person,
                "is_snap": result.metadata.is_snap,
            }
            if show_similarity:
                item["similarity_score"] = result.similarity_score
            output_data.append(item)

        print(json.dumps(output_data, ensure_ascii=False, indent=2))

    elif output_format == "table":
        print(f"\n找到 {len(results)} 个匹配结果:")
        print("-" * 100)

        for i, result in enumerate(results, 1):
            meta = result.metadata
            print(f"{i:2d}. {Path(meta.path).name}")
            print(f"     路径: {meta.path}")
            if meta.description:
                print(
                    f"     描述: {meta.description[:80]}{'...' if len(meta.description) > 80 else ''}"
                )

            tags = []
            if meta.is_landscape:
                tags.append("风景照")
            if meta.has_person:
                tags.append("人物照")
            if meta.is_snap:
                tags.append("截图")
            if tags:
                print(f"     标签: {', '.join(tags)}")

            if show_similarity:
                print(f"     相似度: {result.similarity_score:.3f}")

            print()

    elif output_format == "simple":
        for result in results:
            score_str = f" ({result.similarity_score:.3f})" if show_similarity else ""
            print(f"{result.metadata.path}{score_str}")


def main():
    """主函数"""
    args = parse_arguments()

    # 设置日志
    logger = setup_logging(args.log_level)

    logger.info("=== QwenRag 图片搜索工具启动 ===")

    try:
        # 验证索引目录
        if not Path(args.index_dir).exists():
            logger.error(f"索引目录不存在: {args.index_dir}")
            sys.exit(1)

        # 加载元数据
        logger.info("加载元数据...")
        metadata_list = load_metadata(args.index_dir)
        logger.info(f"加载了 {len(metadata_list)} 条图片元数据")

        # 执行搜索
        results = []

        if args.query:
            # 文本搜索
            logger.info(f"执行文本搜索: '{args.query}'")
            results = text_search(args.query, metadata_list, args.top_k)

        elif args.image_query:
            # 图片搜索
            if not Path(args.image_query).exists():
                logger.error(f"查询图片不存在: {args.image_query}")
                sys.exit(1)

            # 需要API密钥进行图片分析
            api_key = args.api_key or os.getenv("DASHSCOPE_API_KEY")
            if not api_key:
                logger.error("图片搜索需要API密钥 (--api_key 或 DASHSCOPE_API_KEY 环境变量)")
                sys.exit(1)

            logger.info(f"执行图片搜索: '{args.image_query}'")
            qwen_config = QwenVLConfig(api_key=api_key)
            qwen_client = QwenClient(qwen_config=qwen_config)
            processor = ImageProcessor(qwen_client=qwen_client)

            results = image_search(
                args.image_query, metadata_list, processor, args.top_k
            )

        # 输出结果
        format_results(results, args.output_format, args.show_similarity)

        logger.info("搜索完成")

    except KeyboardInterrupt:
        logger.info("用户中断操作")
        sys.exit(1)
    except Exception as e:
        logger.error(f"程序执行出错: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
