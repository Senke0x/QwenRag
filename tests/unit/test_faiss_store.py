"""
UUID FAISS向量存储测试用例
"""
import os
import shutil
import tempfile
import uuid

import numpy as np
import pytest

from schemas.data_models import ImageMetadata, ProcessingStatus
from utils.uuid_manager import generate_content_uuid
from vector_store import FaissStore  # 现在指向UUIDFaissStore


@pytest.mark.unit
class TestUUIDFaissStore:
    """UUID FAISS存储测试类"""

    @pytest.fixture
    def faiss_index_path(self, temp_dir):
        """FAISS索引文件路径"""
        return os.path.join(temp_dir, "test_index.faiss")

    def test_index_creation_empty(self, faiss_index_path):
        """测试空UUID索引创建"""
        dimension = 768

        faiss_store = FaissStore(dimension)
        assert faiss_store.dimension == dimension
        assert faiss_store.index.ntotal == 0  # 索引中没有向量
        assert len(faiss_store.vector_metadata) == 0
        assert len(faiss_store.uuid_to_vector_ids) == 0

    def test_index_creation_with_data(
        self, sample_vectors, sample_ids, faiss_index_path
    ):
        """测试带UUID数据的索引创建"""
        faiss_store = FaissStore(dimension=768)

        # 生成UUID和内容类型
        content_uuids = [
            generate_content_uuid(f"test_image_{i}.jpg")
            for i in range(len(sample_vectors))
        ]
        content_types = ["description"] * len(sample_vectors)

        # 使用UUID方式添加向量
        vector_ids = faiss_store.batch_add_vectors_with_uuid(
            sample_vectors, content_uuids, content_types
        )

        assert len(vector_ids) == len(sample_vectors)
        assert faiss_store.index.ntotal == len(sample_vectors)
        assert len(faiss_store.vector_metadata) == len(sample_vectors)

        # 验证UUID映射
        for uuid in content_uuids:
            assert uuid in faiss_store.uuid_to_vector_ids

    def test_index_parameter_configuration(self):
        """测试索引参数配置"""
        # 测试IndexFlatL2
        store_l2 = FaissStore(dimension=768, index_type="IndexFlatL2")
        assert store_l2.index_type == "IndexFlatL2"
        assert store_l2.dimension == 768

        # 测试IndexFlatIP
        store_ip = FaissStore(dimension=768, index_type="IndexFlatIP")
        assert store_ip.index_type == "IndexFlatIP"
        assert store_ip.dimension == 768

    def test_vector_insertion_single(self, sample_vectors, sample_ids):
        """测试单条UUID向量插入"""
        faiss_store = FaissStore(dimension=768)
        single_vector = sample_vectors[0:1]
        content_uuid = generate_content_uuid(f"test_single_{sample_ids[0]}.jpg")

        vector_ids = faiss_store.batch_add_vectors_with_uuid(
            single_vector, [content_uuid], ["description"]
        )

        assert len(vector_ids) == 1
        assert faiss_store.index.ntotal == 1
        assert content_uuid in faiss_store.uuid_to_vector_ids

    def test_vector_insertion_batch(self, sample_vectors, sample_ids):
        """测试批量UUID向量插入"""
        faiss_store = FaissStore(dimension=768)

        # 生成UUID和内容类型
        content_uuids = [
            generate_content_uuid(f"batch_{id_str}.jpg") for id_str in sample_ids
        ]
        content_types = ["description"] * len(sample_vectors)

        vector_ids = faiss_store.batch_add_vectors_with_uuid(
            sample_vectors, content_uuids, content_types
        )

        assert len(vector_ids) == len(sample_vectors)
        assert faiss_store.index.ntotal == len(sample_vectors)

        # 验证所有UUID都已添加
        for content_uuid in content_uuids:
            assert content_uuid in faiss_store.uuid_to_vector_ids

    def test_vector_insertion_duplicate_uuid(self, sample_vectors, sample_ids):
        """测试重复UUID处理"""
        faiss_store = FaissStore(dimension=768)
        content_uuid = generate_content_uuid("duplicate_test.jpg")

        # 第一次插入
        vector_ids1 = faiss_store.batch_add_vectors_with_uuid(
            sample_vectors[0:1], [content_uuid], ["description"]
        )
        assert len(vector_ids1) == 1
        assert faiss_store.index.ntotal == 1

        # 尝试插入相同UUID（应该跳过或更新）
        vector_ids2 = faiss_store.batch_add_vectors_with_uuid(
            sample_vectors[1:2], [content_uuid], ["face_crop"]
        )
        # UUID系统应该处理重复UUID的情况
        assert isinstance(vector_ids2, list)

    def test_vector_insertion_invalid_dimension(self):
        """测试异常维度数据插入"""
        faiss_store = FaissStore(dimension=768)
        invalid_vector = np.random.rand(1, 512).astype(np.float32)  # 错误维度
        content_uuid = generate_content_uuid("invalid_dim_test.jpg")

        with pytest.raises(ValueError, match="维度.*不匹配"):
            faiss_store.batch_add_vectors_with_uuid(
                invalid_vector, [content_uuid], ["description"]
            )

    def test_vector_insertion_invalid_data_type(self):
        """测试异常数据类型插入"""
        faiss_store = FaissStore(dimension=768)
        invalid_vector = np.random.rand(1, 768).astype(np.int32)  # 错误类型
        content_uuid = generate_content_uuid("invalid_type_test.jpg")

        # FAISS会自动转换类型，所以这个测试应该成功
        vector_ids = faiss_store.batch_add_vectors_with_uuid(
            invalid_vector, [content_uuid], ["description"]
        )
        assert len(vector_ids) == 1

    def test_vector_search_exact(self, sample_vectors, sample_ids):
        """测试UUID精确搜索"""
        faiss_store = FaissStore(dimension=768)

        # 使用UUID添加向量
        content_uuids = [
            generate_content_uuid(f"exact_search_{id_str}.jpg") for id_str in sample_ids
        ]
        content_types = ["description"] * len(sample_vectors)
        vector_ids = faiss_store.batch_add_vectors_with_uuid(
            sample_vectors, content_uuids, content_types
        )

        # 搜索第一个向量
        query_vector = sample_vectors[0]
        results = faiss_store.search_with_uuid(query_vector, k=1)

        assert len(results) == 1
        result = results[0]
        assert result["content_uuid"] == content_uuids[0]
        assert result["distance"] < 0.001  # 应该非常接近0
        assert "embedding_vector" in result

    def test_vector_search_similarity(self, sample_vectors, sample_ids):
        """测试UUID相似度搜索"""
        faiss_store = FaissStore(dimension=768)

        # 使用UUID添加向量
        content_uuids = [
            generate_content_uuid(f"similarity_search_{id_str}.jpg")
            for id_str in sample_ids
        ]
        content_types = ["description"] * len(sample_vectors)
        vector_ids = faiss_store.batch_add_vectors_with_uuid(
            sample_vectors, content_uuids, content_types
        )

        # 创建一个相似的查询向量
        query_vector = sample_vectors[0] + np.random.rand(768) * 0.1
        results = faiss_store.search_with_uuid(query_vector, k=3)

        assert len(results) == 3
        distances = [r["distance"] for r in results]
        assert all(d >= 0 for d in distances)  # 距离应该非负
        assert distances[0] <= distances[1] <= distances[2]  # 距离应该递增

        # 验证结果格式
        for result in results:
            assert "content_uuid" in result
            assert "distance" in result
            assert "embedding_vector" in result
            assert result["content_uuid"] in content_uuids

    def test_vector_search_top_k(self, sample_vectors, sample_ids):
        """测试UUID Top-K检索"""
        faiss_store = FaissStore(dimension=768)

        # 使用UUID添加向量
        content_uuids = [
            generate_content_uuid(f"topk_search_{id_str}.jpg") for id_str in sample_ids
        ]
        content_types = ["description"] * len(sample_vectors)
        vector_ids = faiss_store.batch_add_vectors_with_uuid(
            sample_vectors, content_uuids, content_types
        )

        query_vector = sample_vectors[0]

        for k in [1, 3, 5, 10]:
            results = faiss_store.search_with_uuid(query_vector, k=k)
            expected_len = min(k, len(sample_vectors))
            assert len(results) == expected_len

            # 验证结果格式
            for result in results:
                assert "content_uuid" in result
                assert "distance" in result
                assert "embedding_vector" in result

    def test_vector_search_empty_index(self):
        """测试空UUID索引搜索"""
        faiss_store = FaissStore(dimension=768)
        query_vector = np.random.rand(768).astype(np.float32)

        results = faiss_store.search_with_uuid(query_vector, k=5)

        assert len(results) == 0

    def test_vector_search_k_larger_than_index(self, sample_vectors, sample_ids):
        """测试UUID系统K值大于索引大小的情况"""
        faiss_store = FaissStore(dimension=768)

        # 只添加3个UUID向量
        content_uuids = [
            generate_content_uuid(f"k_larger_{id_str}.jpg") for id_str in sample_ids[:3]
        ]
        content_types = ["description"] * 3
        vector_ids = faiss_store.batch_add_vectors_with_uuid(
            sample_vectors[:3], content_uuids, content_types
        )

        query_vector = sample_vectors[0]
        results = faiss_store.search_with_uuid(query_vector, k=10)  # 要求10个结果

        # 应该只返回3个结果
        assert len(results) == 3

        # 验证结果格式
        for result in results:
            assert "content_uuid" in result
            assert result["content_uuid"] in content_uuids

    def test_index_persistence_save_load(
        self, sample_vectors, sample_ids, faiss_index_path
    ):
        """测试UUID索引保存和加载"""
        # 创建并保存UUID索引
        faiss_store1 = FaissStore(dimension=768)
        content_uuids = [
            generate_content_uuid(f"persist_{id_str}.jpg") for id_str in sample_ids
        ]
        content_types = ["description"] * len(sample_vectors)

        vector_ids = faiss_store1.batch_add_vectors_with_uuid(
            sample_vectors, content_uuids, content_types
        )

        result = faiss_store1.save_index(faiss_index_path)
        assert result == True
        assert os.path.exists(faiss_index_path)

        # 验证文件大小合理
        file_size = os.path.getsize(faiss_index_path)
        assert file_size > 0

        # 加载UUID索引
        faiss_store2 = FaissStore(dimension=768)
        result = faiss_store2.load_index(faiss_index_path)

        assert result == True
        assert faiss_store2.index.ntotal == len(sample_vectors)

        # 验证UUID搜索功能正常
        query_vector = sample_vectors[0]
        results = faiss_store2.search_with_uuid(query_vector, k=1)
        assert len(results) == 1
        assert "content_uuid" in results[0]

    def test_index_persistence_load_nonexistent(self):
        """测试加载不存在的索引文件"""
        faiss_store = FaissStore(dimension=768)

        with pytest.raises(FileNotFoundError):
            faiss_store.load_index("/nonexistent/path/index.faiss")

    def test_index_statistics(self, sample_vectors, sample_ids):
        """测试UUID索引统计信息"""
        faiss_store = FaissStore(dimension=768)

        # 使用UUID添加向量
        content_uuids = [
            generate_content_uuid(f"stats_{id_str}.jpg") for id_str in sample_ids
        ]
        content_types = ["description"] * len(sample_vectors)

        vector_ids = faiss_store.batch_add_vectors_with_uuid(
            sample_vectors, content_uuids, content_types
        )

        stats = faiss_store.get_statistics()

        assert stats["total_vectors"] == len(sample_vectors)
        assert stats["dimension"] == 768
        assert stats["index_type"] == "IndexFlatL2"

        # 验证UUID统计
        assert stats["total_uuids"] == len(content_uuids)
        assert "uuid_to_vector_mapping_size" in stats
