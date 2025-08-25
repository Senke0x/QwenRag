"""
向量存储模块 - 统一使用UUID关联的向量存储
"""

from .uuid_faiss_store import UUIDFaissStore

# 向后兼容：将UUIDFaissStore作为默认的FaissStore和FaceFaissStore
FaissStore = UUIDFaissStore
FaceFaissStore = UUIDFaissStore

__all__ = ["UUIDFaissStore", "FaissStore", "FaceFaissStore"]
