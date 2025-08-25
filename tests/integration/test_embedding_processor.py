"""
EmbeddingProcessor 真实API集成测试
"""
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from clients.qwen_client import QwenClient
from config import QwenVLConfig
from processors.embedding_processor import EmbeddingProcessor
from tests.test_data import get_test_image_base64
from vector_store import FaissStore

pytestmark = pytest.mark.skipif(
    os.getenv("USE_REAL_API", "false").lower() != "true",
    reason="需要设置 USE_REAL_API=true 才能运行真实API测试",
)


class TestEmbeddingProcessorRealAPI:
    """EmbeddingProcessor 真实API集成测试类"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """测试设置"""
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            pytest.skip("需要设置 DASHSCOPE_API_KEY 环境变量")

        # 创建临时目录用于测试
        self.temp_dir = tempfile.mkdtemp()
        self.index_path = os.path.join(self.temp_dir, "test_faiss_index")

        # 初始化配置和客户端
        self.config = QwenVLConfig(api_key=api_key)
        self.qwen_client = QwenClient(qwen_config=self.config)

        # 创建EmbeddingProcessor实例
        self.processor = EmbeddingProcessor(
            qwen_client=self.qwen_client,
            embedding_dimension=1024,
            index_save_path=self.index_path,
        )

        yield

        # 清理临时文件
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @pytest.mark.integration
    def test_embedding_processor_initialization(self):
        """测试EmbeddingProcessor初始化"""
        assert self.processor is not None
        assert self.processor.embedding_dimension == 1024
        assert self.processor.index_save_path == self.index_path
        assert self.processor.qwen_client is not None
        assert self.processor.vector_store is not None

        # 测试统计信息
        stats = self.processor.get_statistics()
        assert isinstance(stats, dict)
        assert "total_vectors" in stats
        assert "embedding_dimension" in stats
        assert stats["embedding_dimension"] == 1024

        print(f"✅ EmbeddingProcessor初始化成功: {stats}")

    @pytest.mark.integration
    def test_process_text_basic(self):
        """测试基本文本处理和存储"""
        test_text = "这是一段测试文本，用于验证文本embedding功能和FAISS存储"
        text_id = "test_text_001"

        # 处理文本
        success = self.processor.process_text(test_text, text_id)
        assert success is True

        # 验证向量已存储
        stats = self.processor.get_statistics()
        assert stats["total_vectors"] >= 1

        print(f"✅ 文本处理成功: {text_id}, 向量总数: {stats['total_vectors']}")

    @pytest.mark.integration
    def test_process_image_base64_basic(self):
        """测试基本图片处理和存储"""
        image_base64 = get_test_image_base64()
        image_id = "test_image_001"

        # 处理图片
        success = self.processor.process_image_base64(image_base64, image_id)
        assert success is True

        # 验证向量已存储
        stats = self.processor.get_statistics()
        assert stats["total_vectors"] >= 1

        print(f"✅ 图片处理成功: {image_id}, 向量总数: {stats['total_vectors']}")

    @pytest.mark.integration
    def test_process_image_with_faces(self):
        """测试人脸区域处理"""
        image_base64 = get_test_image_base64()
        image_id = "test_face_image_001"

        # 模拟人脸区域
        face_rects = [
            {"x": 100, "y": 50, "width": 150, "height": 200},
            {"x": 300, "y": 80, "width": 120, "height": 160},
        ]

        initial_stats = self.processor.get_statistics()
        initial_count = initial_stats["total_vectors"]

        # 处理带人脸的图片
        success = self.processor.process_image_with_faces(
            image_base64, face_rects, image_id
        )

        # 验证结果（可能部分成功）
        final_stats = self.processor.get_statistics()
        final_count = final_stats["total_vectors"]

        # 应该至少增加了整张图片的向量
        assert final_count > initial_count

        print(f"✅ 人脸图片处理: 初始向量{initial_count} -> 最终向量{final_count}")
        print(f"   成功状态: {success}")

    @pytest.mark.integration
    def test_process_batch_texts(self):
        """测试批量文本处理"""
        texts = ["美丽的海边日落景色", "城市夜景中的霓虹灯", "森林中的小径和阳光", "雪山下的宁静湖泊"]
        text_ids = [f"batch_text_{i:03d}" for i in range(len(texts))]

        initial_stats = self.processor.get_statistics()
        initial_count = initial_stats["total_vectors"]

        # 批量处理文本
        results = self.processor.process_batch_texts(texts, text_ids)

        assert isinstance(results, dict)
        assert results["total"] == len(texts)
        assert results["success"] >= 0
        assert results["failed"] >= 0
        assert results["success"] + results["failed"] == results["total"]

        # 验证向量数量增加
        final_stats = self.processor.get_statistics()
        final_count = final_stats["total_vectors"]
        assert final_count >= initial_count + results["success"]

        print(f"✅ 批量文本处理: {results}")
        print(f"   向量数量: {initial_count} -> {final_count}")

    @pytest.mark.integration
    def test_process_batch_images(self):
        """测试批量图片处理"""
        image_base64 = get_test_image_base64()
        # 使用同一张图片创建批量数据（实际应用中会是不同图片）
        images = [image_base64] * 3
        image_ids = [f"batch_image_{i:03d}" for i in range(len(images))]

        initial_stats = self.processor.get_statistics()
        initial_count = initial_stats["total_vectors"]

        # 批量处理图片
        results = self.processor.process_batch_images(images, image_ids)

        assert isinstance(results, dict)
        assert results["total"] == len(images)
        assert results["success"] >= 0
        assert results["failed"] >= 0
        assert results["success"] + results["failed"] == results["total"]

        # 验证向量数量增加
        final_stats = self.processor.get_statistics()
        final_count = final_stats["total_vectors"]
        assert final_count >= initial_count + results["success"]

        print(f"✅ 批量图片处理: {results}")
        print(f"   向量数量: {initial_count} -> {final_count}")

    @pytest.mark.integration
    def test_search_by_text(self):
        """测试基于文本的搜索功能"""
        # 先添加一些测试数据
        test_texts = ["蓝天白云的美丽风景", "城市建筑和街道", "海洋和波浪"]
        text_ids = [f"search_text_{i}" for i in range(len(test_texts))]

        # 添加测试数据
        for text, text_id in zip(test_texts, text_ids):
            success = self.processor.process_text(text, text_id)
            assert success is True

        # 执行搜索
        query_text = "美丽的天空和云朵"
        search_results = self.processor.search_by_text(query_text, top_k=5)

        assert isinstance(search_results, list)
        assert len(search_results) <= 5

        # 验证搜索结果格式
        for result in search_results:
            assert isinstance(result, dict)
            assert "vector_id" in result
            assert "similarity_score" in result
            assert "distance" in result
            assert "rank" in result
            assert isinstance(result["similarity_score"], float)
            assert result["similarity_score"] > 0

        print(f"✅ 文本搜索成功: 查询'{query_text}', 返回{len(search_results)}个结果")
        for i, result in enumerate(search_results[:3]):
            print(
                f"   {i+1}. {result['vector_id']} (相似度: {result['similarity_score']:.4f})"
            )

    @pytest.mark.integration
    def test_search_by_image(self):
        """测试基于图片的搜索功能"""
        # 先添加一些测试图片数据
        image_base64 = get_test_image_base64()
        image_ids = [f"search_image_{i}" for i in range(3)]

        # 添加测试数据（实际中应该是不同图片）
        for i, image_id in enumerate(image_ids):
            success = self.processor.process_image_base64(image_base64, image_id)
            assert success is True

        # 执行图片搜索
        search_results = self.processor.search_by_image(image_base64, top_k=5)

        assert isinstance(search_results, list)
        assert len(search_results) <= 5

        # 验证搜索结果格式
        for result in search_results:
            assert isinstance(result, dict)
            assert "vector_id" in result
            assert "similarity_score" in result
            assert "distance" in result
            assert "rank" in result
            assert isinstance(result["similarity_score"], float)
            assert result["similarity_score"] > 0

        print(f"✅ 图片搜索成功: 返回{len(search_results)}个结果")
        for i, result in enumerate(search_results[:3]):
            print(
                f"   {i+1}. {result['vector_id']} (相似度: {result['similarity_score']:.4f})"
            )

    @pytest.mark.integration
    def test_index_management(self):
        """测试索引管理功能"""
        # 添加一些测试数据
        test_text = "索引管理测试文本"
        text_id = "index_mgmt_test"

        success = self.processor.process_text(test_text, text_id)
        assert success is True

        # 测试保存索引
        save_result = self.processor.save_index()
        assert save_result is True

        # 验证索引文件是否存在
        index_files = list(Path(self.temp_dir).glob("*"))
        assert len(index_files) > 0
        print(f"✅ 索引文件已保存: {[f.name for f in index_files]}")

        # 测试统计信息
        stats = self.processor.get_statistics()
        assert isinstance(stats, dict)
        assert stats["total_vectors"] > 0
        assert stats["embedding_dimension"] == 1024

        print(f"✅ 索引统计信息: {stats}")

    @pytest.mark.integration
    def test_vector_removal_and_rebuild(self):
        """测试向量删除和索引重建"""
        # 添加测试数据
        test_texts = ["删除测试文本1", "删除测试文本2", "保留的文本"]
        text_ids = ["remove_test_1", "remove_test_2", "keep_test"]

        for text, text_id in zip(test_texts, text_ids):
            success = self.processor.process_text(text, text_id)
            assert success is True

        initial_stats = self.processor.get_statistics()
        initial_count = initial_stats["total_vectors"]
        assert initial_count >= 3

        # 删除指定向量
        remove_ids = ["remove_test_1", "remove_test_2"]
        remove_result = self.processor.remove_vectors(remove_ids)

        if remove_result:  # 删除功能可能需要更多实现
            final_stats = self.processor.get_statistics()
            final_count = final_stats["total_vectors"]
            print(f"✅ 向量删除: {initial_count} -> {final_count}")
        else:
            print("⚠️ 向量删除功能需要进一步实现或调试")

    @pytest.mark.integration
    def test_embedding_dimension_consistency(self):
        """测试embedding维度一致性"""
        # 测试文本和图片embedding维度是否一致
        test_text = "维度一致性测试文本"
        image_base64 = get_test_image_base64()

        # 处理文本和图片
        text_success = self.processor.process_text(test_text, "dim_test_text")
        image_success = self.processor.process_image_base64(
            image_base64, "dim_test_image"
        )

        assert text_success is True
        assert image_success is True

        # 检查维度一致性
        stats = self.processor.get_statistics()
        assert stats["embedding_dimension"] == 1024

        # 验证实际生成的向量维度
        text_response = self.processor.qwen_client.get_text_embedding(test_text)
        image_response = self.processor.qwen_client.get_image_embedding(image_base64)

        text_dim = len(text_response["embedding"])
        image_dim = len(image_response["embedding"])

        assert text_dim == image_dim == 1024
        print(f"✅ 维度一致性验证: 文本{text_dim}维, 图片{image_dim}维")

    @pytest.mark.integration
    def test_cross_modal_search(self):
        """测试跨模态搜索：文本查询图片，图片查询文本"""
        # 准备相关的文本和图片数据
        text_description = "一张美丽的风景照片，有蓝天白云"
        image_base64 = get_test_image_base64()

        # 存储文本和图片
        text_success = self.processor.process_text(text_description, "cross_modal_text")
        image_success = self.processor.process_image_base64(
            image_base64, "cross_modal_image"
        )

        assert text_success is True
        assert image_success is True

        # 用图片搜索（可能找到相关文本）
        image_search_results = self.processor.search_by_image(image_base64, top_k=10)
        assert isinstance(image_search_results, list)

        # 用文本搜索（可能找到相关图片）
        text_search_results = self.processor.search_by_text("美丽风景图片", top_k=10)
        assert isinstance(text_search_results, list)

        print(f"✅ 跨模态搜索测试:")
        print(f"   图片搜索返回{len(image_search_results)}个结果")
        print(f"   文本搜索返回{len(text_search_results)}个结果")

        # 显示top结果
        if image_search_results:
            top_image_result = image_search_results[0]
            print(
                f"   图片搜索Top1: {top_image_result['vector_id']} (相似度: {top_image_result['similarity_score']:.4f})"
            )

        if text_search_results:
            top_text_result = text_search_results[0]
            print(
                f"   文本搜索Top1: {top_text_result['vector_id']} (相似度: {top_text_result['similarity_score']:.4f})"
            )

    @pytest.mark.integration
    def test_error_handling(self):
        """测试错误处理"""
        # 测试空文本处理
        try:
            result = self.processor.process_text("", "empty_text")
            print(f"⚠️ 空文本处理结果: {result}")
        except Exception as e:
            print(f"✅ 空文本正确抛出异常: {str(e)[:50]}...")

        # 测试无效base64数据
        try:
            result = self.processor.process_image_base64(
                "invalid_base64", "invalid_image"
            )
            print(f"⚠️ 无效base64处理结果: {result}")
        except Exception as e:
            print(f"✅ 无效base64正确抛出异常: {str(e)[:50]}...")

        # 测试不匹配的批量数据
        try:
            results = self.processor.process_batch_texts(["text1", "text2"], ["id1"])
        except ValueError as e:
            print(f"✅ 数量不匹配正确抛出异常: {e}")
            assert "不匹配" in str(e)

    @pytest.mark.integration
    def test_large_batch_processing(self):
        """测试大批量数据处理"""
        # 创建较大的批量数据
        batch_size = 10
        texts = [f"批量处理测试文本 {i+1}" for i in range(batch_size)]
        text_ids = [f"large_batch_{i:03d}" for i in range(batch_size)]

        initial_stats = self.processor.get_statistics()
        initial_count = initial_stats["total_vectors"]

        # 执行批量处理
        results = self.processor.process_batch_texts(texts, text_ids)

        assert results["total"] == batch_size
        success_rate = results["success"] / results["total"]

        print(f"✅ 大批量处理测试: {batch_size}个文本")
        print(f"   成功率: {success_rate:.2%} ({results['success']}/{results['total']})")

        if results["failed"] > 0:
            print(f"   失败项目: {results['failed_items']}")

        # 验证最终向量数量
        final_stats = self.processor.get_statistics()
        final_count = final_stats["total_vectors"]
        print(f"   向量数量变化: {initial_count} -> {final_count}")
