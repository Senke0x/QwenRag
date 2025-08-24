"""
FAISS向量存储测试用例
"""
import os
import shutil
import tempfile

import numpy as np
import pytest

from schemas.data_models import ImageMetadata, ProcessingStatus
from vector_store.faiss_store import FaissStore


@pytest.mark.unit
class TestFaissStore:
    """FAISS存储测试类"""

    @pytest.fixture
    def faiss_index_path(self, temp_dir):
        """FAISS索引文件路径"""
        return os.path.join(temp_dir, "test_index.faiss")

    def test_index_creation_empty(self, faiss_index_path):
        """测试空索引创建"""
        dimension = 768

        faiss_store = FaissStore(dimension)
        assert faiss_store.dimension == dimension
        assert faiss_store.index.ntotal == 0  # 索引中没有向量
        assert len(faiss_store.id_mapping) == 0
        assert len(faiss_store.reverse_id_mapping) == 0

    def test_index_creation_with_data(
        self, sample_vectors, sample_ids, faiss_index_path
    ):
        """测试带数据的索引创建"""
        faiss_store = FaissStore(dimension=768)
        success = faiss_store.add_vectors(sample_vectors, sample_ids)

        assert success == True
        assert faiss_store.index.ntotal == len(sample_vectors)
        assert len(faiss_store.id_mapping) == len(sample_ids)

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
        """测试单条向量插入"""
        faiss_store = FaissStore(dimension=768)
        single_vector = sample_vectors[0:1]
        single_id = [sample_ids[0]]

        result = faiss_store.add_vectors(single_vector, single_id)

        assert result == True
        assert faiss_store.index.ntotal == 1
        assert single_id[0] in faiss_store.reverse_id_mapping

    def test_vector_insertion_batch(self, sample_vectors, sample_ids):
        """测试批量向量插入"""
        faiss_store = FaissStore(dimension=768)
        result = faiss_store.add_vectors(sample_vectors, sample_ids)

        assert result == True
        assert faiss_store.index.ntotal == len(sample_vectors)

        # 验证所有ID都已添加
        for id_str in sample_ids:
            assert id_str in faiss_store.reverse_id_mapping

    def test_vector_insertion_duplicate_id(self, sample_vectors, sample_ids):
        """测试重复ID处理"""
        faiss_store = FaissStore(dimension=768)

        # 第一次插入
        result1 = faiss_store.add_vectors(sample_vectors[0:1], [sample_ids[0]])
        assert result1 == True
        assert faiss_store.index.ntotal == 1

        # 尝试插入相同ID（默认更新策略）
        result2 = faiss_store.add_vectors(
            sample_vectors[1:2], [sample_ids[0]], update_existing=True
        )
        assert result2 == True
        # FAISS不支持原地更新，但函数应该返回True

    def test_vector_insertion_invalid_dimension(self):
        """测试异常维度数据插入"""
        faiss_store = FaissStore(dimension=768)
        invalid_vector = np.random.rand(1, 512).astype(np.float32)  # 错误维度

        with pytest.raises(ValueError, match="维度.*不匹配"):
            faiss_store.add_vectors(invalid_vector, ["test_id"])

    def test_vector_insertion_invalid_data_type(self):
        """测试异常数据类型插入"""
        faiss_store = FaissStore(dimension=768)
        invalid_vector = np.random.rand(1, 768).astype(np.int32)  # 错误类型

        # FAISS会自动转换类型，所以这个测试应该成功
        result = faiss_store.add_vectors(invalid_vector, ["test_id"])
        assert result == True

    def test_vector_search_exact(self, sample_vectors, sample_ids):
        """测试精确搜索"""
        faiss_store = FaissStore(dimension=768)
        faiss_store.add_vectors(sample_vectors, sample_ids)

        # 搜索第一个向量
        query_vector = sample_vectors[0:1]
        distances, indices, ids = faiss_store.search(query_vector, k=1)

        assert len(indices) == 1
        assert ids[0] == sample_ids[0]
        assert distances[0] < 0.001  # 应该非常接近0

    def test_vector_search_similarity(self, sample_vectors, sample_ids):
        """测试相似度搜索"""
        faiss_store = FaissStore(dimension=768)
        faiss_store.add_vectors(sample_vectors, sample_ids)

        # 创建一个相似的查询向量
        query_vector = sample_vectors[0:1] + np.random.rand(1, 768) * 0.1
        distances, indices, ids = faiss_store.search(query_vector, k=3)

        assert len(indices) == 3
        assert len(distances) == 3
        assert all(d >= 0 for d in distances)  # 距离应该非负
        assert distances[0] <= distances[1] <= distances[2]  # 距离应该递增

    def test_vector_search_top_k(self, sample_vectors, sample_ids):
        """测试Top-K检索"""
        faiss_store = FaissStore(dimension=768)
        faiss_store.add_vectors(sample_vectors, sample_ids)

        query_vector = sample_vectors[0:1]

        for k in [1, 3, 5, 10]:
            distances, indices, ids = faiss_store.search(query_vector, k=k)
            expected_len = min(k, len(sample_vectors))
            assert len(indices) == expected_len
            assert len(distances) == expected_len
            assert len(ids) == expected_len

    def test_vector_search_empty_index(self):
        """测试空索引搜索"""
        faiss_store = FaissStore(dimension=768)
        query_vector = np.random.rand(1, 768).astype(np.float32)

        distances, indices, ids = faiss_store.search(query_vector, k=5)

        assert len(indices) == 0
        assert len(distances) == 0
        assert len(ids) == 0

    def test_vector_search_k_larger_than_index(self, sample_vectors, sample_ids):
        """测试K值大于索引大小的情况"""
        faiss_store = FaissStore(dimension=768)
        faiss_store.add_vectors(sample_vectors[:3], sample_ids[:3])  # 只添加3个向量

        query_vector = sample_vectors[0:1]
        distances, indices, ids = faiss_store.search(query_vector, k=10)  # 要求10个结果

        # 应该只返回3个结果
        assert len(indices) == 3
        assert len(distances) == 3
        assert len(ids) == 3

    def test_index_persistence_save_load(
        self, sample_vectors, sample_ids, faiss_index_path
    ):
        """测试索引保存和加载"""
        # 创建并保存索引
        faiss_store1 = FaissStore(dimension=768)
        faiss_store1.add_vectors(sample_vectors, sample_ids)

        result = faiss_store1.save_index(faiss_index_path)
        assert result == True
        assert os.path.exists(faiss_index_path)

        # 验证文件大小合理
        file_size = os.path.getsize(faiss_index_path)
        assert file_size > 0

        # 加载索引
        faiss_store2 = FaissStore(dimension=768)
        result = faiss_store2.load_index(faiss_index_path)

        assert result == True
        assert faiss_store2.index.ntotal == len(sample_vectors)

        # 验证搜索功能正常
        query_vector = sample_vectors[0:1]
        distances, indices, ids = faiss_store2.search(query_vector, k=1)
        assert ids[0] == sample_ids[0]

    def test_index_persistence_load_nonexistent(self):
        """测试加载不存在的索引文件"""
        faiss_store = FaissStore(dimension=768)

        with pytest.raises(FileNotFoundError):
            faiss_store.load_index("/nonexistent/path/index.faiss")

    def test_index_statistics(self, sample_vectors, sample_ids):
        """测试索引统计信息"""
        faiss_store = FaissStore(dimension=768)
        faiss_store.add_vectors(sample_vectors, sample_ids)

        stats = faiss_store.get_statistics()

        assert stats["total_vectors"] == len(sample_vectors)
        assert stats["dimension"] == 768
        assert stats["index_type"] == "IndexFlatL2"
