"""
Pipeline集成测试 - 基于真实API
测试完整的端到端流程，包括索引构建和检索查询
"""
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest

from clients.qwen_client import QwenClient
from pipelines.indexing_pipeline import IndexingPipeline
from pipelines.retrieval_pipeline import RetrievalPipeline
from processors.embedding_processor import EmbeddingProcessor
from processors.image_processor import ImageProcessor


@pytest.mark.integration
class TestPipelineIntegrationRealAPI:
    """Pipeline集成测试类 - 使用真实API"""

    @pytest.fixture(scope="class")
    def api_key_check(self):
        """检查API密钥是否可用"""
        api_key = os.getenv("DASHSCOPE_API_KEY")
        use_real_api = os.getenv("USE_REAL_API", "false").lower() == "true"

        if not use_real_api or not api_key:
            pytest.skip("跳过真实API测试：需要设置USE_REAL_API=true和DASHSCOPE_API_KEY")

        return api_key

    @pytest.fixture(scope="class")
    def test_images_setup(self):
        """准备测试图片"""
        test_dir = tempfile.mkdtemp(prefix="pipeline_test_")

        # 从dataset目录复制几张测试图片
        dataset_dir = Path("dataset")
        if dataset_dir.exists():
            image_files = list(dataset_dir.glob("*.jpg"))[:3]  # 只取3张图片

            for i, image_file in enumerate(image_files):
                dest_file = Path(test_dir) / f"test_image_{i+1}.jpg"
                shutil.copy2(image_file, dest_file)

        yield test_dir

        # 清理测试目录
        shutil.rmtree(test_dir, ignore_errors=True)

    @pytest.fixture(scope="class")
    def temp_index_paths(self):
        """临时索引文件路径"""
        temp_dir = tempfile.mkdtemp(prefix="pipeline_index_")

        paths = {
            "metadata_path": Path(temp_dir) / "test_metadata.json",
            "index_path": Path(temp_dir) / "test_index.faiss",
            "temp_dir": temp_dir,
        }

        yield paths

        # 清理
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_indexing_pipeline_initialization_real_api(self, api_key_check):
        """测试IndexingPipeline初始化 - 真实API"""
        try:
            pipeline = IndexingPipeline(
                batch_size=2, max_workers=1, auto_save=True  # 降低并发避免API限制
            )

            assert pipeline is not None
            assert pipeline.batch_size == 2
            assert pipeline.max_workers == 1
            assert pipeline.auto_save is True

            # 验证组件初始化
            assert isinstance(pipeline.image_processor, ImageProcessor)
            assert isinstance(pipeline.embedding_processor, EmbeddingProcessor)
            assert isinstance(pipeline.image_processor.qwen_client, QwenClient)

        except Exception as e:
            pytest.fail(f"IndexingPipeline初始化失败: {e}")

    def test_indexing_pipeline_scan_directory_real_api(
        self, api_key_check, test_images_setup
    ):
        """测试目录扫描功能 - 真实API"""
        pipeline = IndexingPipeline()

        # 扫描测试目录
        image_paths = pipeline.scan_image_directory(test_images_setup)

        assert len(image_paths) > 0, "应该找到测试图片"

        # 验证路径有效性
        for path in image_paths:
            assert Path(path).exists(), f"图片文件不存在: {path}"
            assert Path(path).suffix.lower() in [
                ".jpg",
                ".jpeg",
                ".png",
            ], f"不支持的图片格式: {path}"

    def test_indexing_pipeline_process_single_image_real_api(
        self, api_key_check, test_images_setup, temp_index_paths
    ):
        """测试单张图片处理 - 真实API"""
        pipeline = IndexingPipeline(
            metadata_save_path=str(temp_index_paths["metadata_path"]),
            batch_size=1,
            max_workers=1,
            auto_save=False,  # 手动控制保存
        )

        # 获取第一张测试图片
        image_paths = pipeline.scan_image_directory(test_images_setup)
        assert len(image_paths) > 0, "没有找到测试图片"

        test_image = image_paths[0]

        # 处理单张图片
        success, result = pipeline._process_single_image_complete(test_image)

        # 验证处理结果
        assert success is True, f"图片处理失败: {result.get('error', 'Unknown error')}"
        assert result["status"] == "success"
        assert result["image_path"] == test_image
        assert "unique_id" in result
        assert "embedding_results" in result
        assert "embedding_stats" in result

        # 验证embedding结果
        embedding_results = result["embedding_results"]
        assert "full_image" in embedding_results
        assert embedding_results["full_image"] is True, "全图embedding应该成功"

        if result["metadata"].get("description"):
            assert "description" in embedding_results

        # 验证统计信息
        stats = result["embedding_stats"]
        assert stats["successful"] > 0, "应该有成功的embedding"
        assert stats["total"] > 0, "应该有总的embedding尝试"

    def test_indexing_pipeline_batch_processing_real_api(
        self, api_key_check, test_images_setup, temp_index_paths
    ):
        """测试批量处理 - 真实API"""
        pipeline = IndexingPipeline(
            metadata_save_path=str(temp_index_paths["metadata_path"]),
            batch_size=2,
            max_workers=1,  # 串行处理避免API限制
            auto_save=True,
        )

        # 获取测试图片
        image_paths = pipeline.scan_image_directory(test_images_setup)
        test_images = image_paths[:2]  # 只处理前两张

        if len(test_images) == 0:
            pytest.skip("没有测试图片可用")

        # 批量处理
        start_time = time.time()
        results = pipeline.process_image_batch(test_images, parallel=False)  # 串行避免API限制
        processing_time = time.time() - start_time

        # 验证处理结果
        assert results["total"] == len(test_images)
        assert results["success"] > 0, "应该有成功处理的图片"
        assert results["failed"] + results["success"] == results["total"]

        # 验证元数据保存
        assert temp_index_paths["metadata_path"].exists(), "元数据文件应该被创建"

        with open(temp_index_paths["metadata_path"], "r", encoding="utf-8") as f:
            metadata = json.load(f)

        assert len(metadata) == results["success"], "保存的元数据数量应该等于成功处理的数量"

        # 验证统计信息
        stats = pipeline.get_statistics()
        assert stats["success_count"] == results["success"]
        assert stats["total_processed"] == results["total"]

        print(
            f"批量处理完成: {results['success']}/{results['total']} 成功, 用时: {processing_time:.2f}秒"
        )

    def test_retrieval_pipeline_initialization_real_api(
        self, api_key_check, temp_index_paths
    ):
        """测试RetrievalPipeline初始化 - 真实API"""
        # 创建一个空的元数据文件用于测试
        temp_index_paths["metadata_path"].parent.mkdir(parents=True, exist_ok=True)
        with open(temp_index_paths["metadata_path"], "w", encoding="utf-8") as f:
            json.dump([], f)

        try:
            pipeline = RetrievalPipeline(
                metadata_path=str(temp_index_paths["metadata_path"]),
                default_top_k=5,
                similarity_threshold=0.3,
            )

            assert pipeline is not None
            assert pipeline.default_top_k == 5
            assert pipeline.similarity_threshold == 0.3
            assert isinstance(pipeline.embedding_processor, EmbeddingProcessor)

        except Exception as e:
            pytest.fail(f"RetrievalPipeline初始化失败: {e}")

    def test_retrieval_pipeline_text_search_real_api(self, api_key_check):
        """测试文本搜索功能 - 真实API"""
        # 创建一个包含测试数据的元数据文件
        temp_dir = tempfile.mkdtemp()
        metadata_path = Path(temp_dir) / "test_metadata.json"

        try:
            pipeline = RetrievalPipeline(
                metadata_path=str(metadata_path), default_top_k=3
            )

            # 测试空查询
            results = pipeline.search_by_text("")
            assert results == [], "空查询应该返回空结果"

            # 测试正常查询（虽然没有数据，但应该正常执行）
            results = pipeline.search_by_text("测试查询")
            assert isinstance(results, list), "应该返回列表"

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.slow
    def test_end_to_end_pipeline_flow_real_api(
        self, api_key_check, test_images_setup, temp_index_paths
    ):
        """测试完整的端到端流程 - 真实API"""
        print("\n开始端到端流程测试...")

        # Step 1: 索引构建
        print("Step 1: 构建索引...")
        indexing_pipeline = IndexingPipeline(
            metadata_save_path=str(temp_index_paths["metadata_path"]),
            batch_size=2,
            max_workers=1,
            auto_save=True,
        )

        # 获取测试图片
        image_paths = indexing_pipeline.scan_image_directory(test_images_setup)
        if len(image_paths) == 0:
            pytest.skip("没有测试图片可用")

        # 限制处理数量以节省API调用
        test_images = image_paths[:2]

        # 构建索引
        start_time = time.time()
        indexing_results = indexing_pipeline.build_index_from_directory(
            test_images_setup, recursive=False, parallel=False  # 串行处理避免API限制
        )
        indexing_time = time.time() - start_time

        print(f"索引构建完成: {indexing_results['success']}/{indexing_results['total']} 成功")
        print(f"索引构建用时: {indexing_time:.2f}秒")

        # 验证索引构建结果
        assert indexing_results["success"] > 0, "应该有成功构建的索引"
        assert temp_index_paths["metadata_path"].exists(), "元数据文件应该存在"

        # Step 2: 检索测试
        print("Step 2: 测试检索...")
        retrieval_pipeline = RetrievalPipeline(
            metadata_path=str(temp_index_paths["metadata_path"]),
            default_top_k=5,
            similarity_threshold=0.1,  # 降低阈值以获得更多结果
        )

        # 验证元数据加载
        assert len(retrieval_pipeline.metadata_index) > 0, "应该加载到元数据"

        # 测试文本搜索
        text_queries = ["游戏", "图片", "截图"]
        for query in text_queries:
            print(f"测试文本查询: '{query}'")
            results = retrieval_pipeline.search_by_text(query, top_k=3)
            print(f"  返回结果数: {len(results)}")

            for result in results:
                assert "unique_id" in result
                assert "similarity_score" in result
                assert "image_path" in result
                assert Path(result["image_path"]).exists()

        # 测试图片搜索
        if test_images:
            print(f"测试图片查询: {Path(test_images[0]).name}")
            image_results = retrieval_pipeline.search_by_image(test_images[0], top_k=3)
            print(f"  返回结果数: {len(image_results)}")

            for result in image_results:
                assert "unique_id" in result
                assert "similarity_score" in result

        # Step 3: 统计验证
        print("Step 3: 验证统计信息...")
        indexing_stats = indexing_pipeline.get_statistics()
        retrieval_stats = retrieval_pipeline.get_statistics()

        print(f"索引统计:")
        print(f"  成功处理: {indexing_stats['success_count']}")
        print(f"  失败处理: {indexing_stats['failed_count']}")
        print(f"  元数据数: {indexing_stats['metadata_count']}")

        print(f"检索统计:")
        print(f"  向量数量: {retrieval_stats['vector_count']}")
        print(f"  元数据数: {retrieval_stats['metadata_count']}")
        print(f"  向量维度: {retrieval_stats['index_dimension']}")

        # 验证统计一致性
        assert indexing_stats["success_count"] == retrieval_stats["metadata_count"]
        assert retrieval_stats["vector_count"] > 0
        assert retrieval_stats["index_dimension"] > 0

        print("✅ 端到端流程测试完成!")

    def test_retrieval_pipeline_metadata_enhancement_real_api(
        self, api_key_check, temp_index_paths
    ):
        """测试检索结果的元数据增强 - 真实API"""
        # 创建测试元数据
        test_metadata = [
            {
                "unique_id": "test_img_001",
                "image_path": "/test/image1.jpg",
                "status": "success",
                "metadata": {
                    "description": "测试图片1",
                    "is_snap": True,
                    "has_person": False,
                },
            }
        ]

        temp_index_paths["metadata_path"].parent.mkdir(parents=True, exist_ok=True)
        with open(temp_index_paths["metadata_path"], "w", encoding="utf-8") as f:
            json.dump(test_metadata, f)

        pipeline = RetrievalPipeline(
            metadata_path=str(temp_index_paths["metadata_path"])
        )

        # 测试unique_id提取
        assert (
            pipeline._extract_unique_id_from_vector_id("test_img_001_full")
            == "test_img_001"
        )
        assert (
            pipeline._extract_unique_id_from_vector_id("test_img_001_desc")
            == "test_img_001"
        )
        assert (
            pipeline._extract_unique_id_from_vector_id("test_img_001_face_0")
            == "test_img_001"
        )

        # 测试匹配类型判断
        assert pipeline._determine_match_type("test_img_001_full") == "image_full"
        assert pipeline._determine_match_type("test_img_001_desc") == "description"
        assert pipeline._determine_match_type("test_img_001_face_0") == "face"

        # 验证元数据加载
        assert len(pipeline.metadata_index) == 1
        assert "test_img_001" in pipeline.metadata_index

    def test_pipeline_error_handling_real_api(self, api_key_check):
        """测试pipeline错误处理 - 真实API"""
        # 测试不存在的目录
        pipeline = IndexingPipeline()

        with pytest.raises(ValueError, match="目录不存在"):
            pipeline.scan_image_directory("/nonexistent/directory")

        # 测试不存在的元数据文件
        retrieval = RetrievalPipeline(metadata_path="/nonexistent/metadata.json")
        assert len(retrieval.metadata_index) == 0

        # 测试不存在的图片文件
        results = retrieval.search_by_image("/nonexistent/image.jpg")
        assert results == []

    @pytest.mark.slow
    def test_pipeline_performance_real_api(self, api_key_check, test_images_setup):
        """测试pipeline性能 - 真实API"""
        pipeline = IndexingPipeline(batch_size=1, max_workers=1, auto_save=False)

        image_paths = pipeline.scan_image_directory(test_images_setup)
        if len(image_paths) == 0:
            pytest.skip("没有测试图片可用")

        # 只测试一张图片以节省API调用
        test_image = image_paths[0]

        # 测量处理时间
        start_time = time.time()
        success, result = pipeline._process_single_image_complete(test_image)
        processing_time = time.time() - start_time

        print(f"单张图片处理时间: {processing_time:.2f}秒")

        # 基本性能检查（不要设置太严格的限制）
        assert processing_time < 60, "单张图片处理时间不应超过60秒"
        assert success is True, "图片处理应该成功"

        # 检查结果的完整性
        assert "embedding_stats" in result
        stats = result["embedding_stats"]
        assert stats["successful"] > 0
        assert stats["total"] > 0
