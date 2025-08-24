"""
结构化数据缓存管理 - 使用DiskCache存储Qwen解析的结构化数据
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from diskcache import Cache
except ImportError:
    raise ImportError("需要安装diskcache: pip install diskcache")

from utils.logger import setup_logger

logger = setup_logger(__name__)


class StructuredDataCache:
    """结构化数据缓存管理器"""

    def __init__(
        self,
        cache_dir: str = "data/structured_cache",
        size_limit: int = 10**9,  # 1GB
        eviction_policy: str = "lru",
    ):
        """
        初始化结构化数据缓存

        Args:
            cache_dir: 缓存目录路径
            size_limit: 缓存大小限制（字节）
            eviction_policy: 缓存淘汰策略 ('lru', 'lfu', 'fifo')
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 初始化DiskCache
        self.cache = Cache(
            str(self.cache_dir), size_limit=size_limit, eviction_policy=eviction_policy
        )

        logger.info(f"StructuredDataCache初始化完成, 缓存目录: {self.cache_dir}")

    def store_analysis_result(self, content_uuid: str, data: Dict[str, Any]) -> bool:
        """
        存储Qwen分析结果

        Args:
            content_uuid: 内容UUID
            data: 要存储的数据字典

        Returns:
            是否存储成功
        """
        try:
            # 添加存储时间戳
            enhanced_data = {
                **data,
                "cache_timestamp": datetime.now().isoformat(),
                "uuid": content_uuid,
            }

            # 存储到DiskCache
            self.cache[content_uuid] = enhanced_data

            logger.debug(f"存储分析结果成功: UUID={content_uuid}")
            return True

        except Exception as e:
            logger.error(f"存储分析结果失败: UUID={content_uuid}, 错误: {e}")
            return False

    def get_analysis_result(self, content_uuid: str) -> Optional[Dict[str, Any]]:
        """
        根据UUID获取分析结果

        Args:
            content_uuid: 内容UUID

        Returns:
            分析结果字典，如果不存在返回None
        """
        try:
            result = self.cache.get(content_uuid)
            if result:
                logger.debug(f"获取分析结果成功: UUID={content_uuid}")
                return result
            else:
                logger.debug(f"分析结果不存在: UUID={content_uuid}")
                return None

        except Exception as e:
            logger.error(f"获取分析结果失败: UUID={content_uuid}, 错误: {e}")
            return None

    def batch_store(
        self, uuid_data_pairs: List[Tuple[str, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        批量存储分析结果

        Args:
            uuid_data_pairs: (UUID, 数据)元组列表

        Returns:
            批量操作结果统计
        """
        success_count = 0
        failed_count = 0
        failed_uuids = []

        for content_uuid, data in uuid_data_pairs:
            if self.store_analysis_result(content_uuid, data):
                success_count += 1
            else:
                failed_count += 1
                failed_uuids.append(content_uuid)

        result = {
            "total": len(uuid_data_pairs),
            "success": success_count,
            "failed": failed_count,
            "failed_uuids": failed_uuids,
        }

        logger.info(f"批量存储完成: {result}")
        return result

    def batch_get(
        self, content_uuids: List[str]
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        批量获取分析结果

        Args:
            content_uuids: UUID列表

        Returns:
            UUID到数据的映射字典
        """
        results = {}

        for content_uuid in content_uuids:
            results[content_uuid] = self.get_analysis_result(content_uuid)

        found_count = sum(1 for result in results.values() if result is not None)
        logger.debug(f"批量获取完成: 请求{len(content_uuids)}个, 找到{found_count}个")

        return results

    def exists(self, content_uuid: str) -> bool:
        """
        检查UUID对应的数据是否存在

        Args:
            content_uuid: 内容UUID

        Returns:
            是否存在
        """
        try:
            return content_uuid in self.cache
        except Exception as e:
            logger.error(f"检查存在性失败: UUID={content_uuid}, 错误: {e}")
            return False

    def delete(self, content_uuid: str) -> bool:
        """
        删除指定UUID的数据

        Args:
            content_uuid: 内容UUID

        Returns:
            是否删除成功
        """
        try:
            if content_uuid in self.cache:
                del self.cache[content_uuid]
                logger.debug(f"删除数据成功: UUID={content_uuid}")
                return True
            else:
                logger.debug(f"数据不存在，无需删除: UUID={content_uuid}")
                return True

        except Exception as e:
            logger.error(f"删除数据失败: UUID={content_uuid}, 错误: {e}")
            return False

    def clear_all(self) -> bool:
        """
        清空所有缓存数据

        Returns:
            是否清空成功
        """
        try:
            self.cache.clear()
            logger.info("清空所有缓存数据成功")
            return True

        except Exception as e:
            logger.error(f"清空缓存失败: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取缓存统计信息

        Returns:
            统计信息字典
        """
        try:
            stats = {
                "total_items": len(self.cache),
                "cache_size_bytes": self.cache.volume(),
                "cache_directory": str(self.cache_dir),
                "eviction_policy": self.cache.eviction_policy,
                "size_limit": self.cache.size_limit,
            }

            return stats

        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {}

    def list_all_uuids(self) -> List[str]:
        """
        获取所有已存储的UUID列表

        Returns:
            UUID列表
        """
        try:
            return list(self.cache.iterkeys())
        except Exception as e:
            logger.error(f"获取UUID列表失败: {e}")
            return []

    def update_analysis_result(
        self, content_uuid: str, updates: Dict[str, Any]
    ) -> bool:
        """
        更新已存储的分析结果

        Args:
            content_uuid: 内容UUID
            updates: 要更新的字段字典

        Returns:
            是否更新成功
        """
        try:
            existing_data = self.get_analysis_result(content_uuid)
            if existing_data is None:
                logger.warning(f"要更新的数据不存在: UUID={content_uuid}")
                return False

            # 合并更新
            updated_data = {**existing_data, **updates}
            updated_data["last_updated"] = datetime.now().isoformat()

            return self.store_analysis_result(content_uuid, updated_data)

        except Exception as e:
            logger.error(f"更新分析结果失败: UUID={content_uuid}, 错误: {e}")
            return False

    def search_by_field(
        self, field_name: str, field_value: Any, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        根据字段值搜索数据

        Args:
            field_name: 字段名
            field_value: 字段值
            limit: 结果数量限制

        Returns:
            匹配的数据列表
        """
        try:
            results = []
            count = 0

            for key in self.cache.iterkeys():
                if count >= limit:
                    break

                data = self.cache.get(key)
                if data and data.get(field_name) == field_value:
                    results.append(data)
                    count += 1

            logger.debug(f"字段搜索完成: {field_name}={field_value}, 找到{len(results)}条结果")
            return results

        except Exception as e:
            logger.error(f"字段搜索失败: {field_name}={field_value}, 错误: {e}")
            return []

    def close(self):
        """关闭缓存"""
        try:
            self.cache.close()
            logger.info("StructuredDataCache已关闭")
        except Exception as e:
            logger.error(f"关闭缓存失败: {e}")


# 全局缓存实例
_global_cache: Optional[StructuredDataCache] = None


def get_global_cache() -> StructuredDataCache:
    """
    获取全局缓存实例（单例模式）

    Returns:
        全局StructuredDataCache实例
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = StructuredDataCache()
    return _global_cache


def reset_global_cache():
    """重置全局缓存实例"""
    global _global_cache
    if _global_cache is not None:
        _global_cache.close()
        _global_cache = None
