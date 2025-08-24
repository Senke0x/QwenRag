"""
UUID管理工具 - 为内容生成和管理唯一标识符
"""
import hashlib
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional


class UUIDManager:
    """UUID管理器 - 负责生成和管理内容UUID"""

    def __init__(self, deterministic: bool = False):
        """
        初始化UUID管理器

        Args:
            deterministic: 是否使用确定性UUID生成（基于内容哈希）
        """
        self.deterministic = deterministic

    def generate_content_uuid(self, image_path: str) -> str:
        """
        为图片内容生成UUID

        Args:
            image_path: 图片路径

        Returns:
            UUID字符串
        """
        if self.deterministic:
            return self._generate_deterministic_uuid(image_path)
        else:
            return self._generate_random_uuid()

    def _generate_deterministic_uuid(self, image_path: str) -> str:
        """
        基于文件内容生成确定性UUID

        Args:
            image_path: 图片路径

        Returns:
            确定性UUID字符串
        """
        try:
            file_path = Path(image_path)
            if not file_path.exists():
                raise FileNotFoundError(f"文件不存在: {image_path}")

            # 使用文件内容和路径生成哈希
            content_hash = hashlib.sha256()

            # 添加文件内容
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    content_hash.update(chunk)

            # 添加文件路径信息（标准化后的绝对路径）
            content_hash.update(str(file_path.resolve()).encode("utf-8"))

            # 使用SHA256哈希生成确定性UUID
            hash_hex = content_hash.hexdigest()
            return str(uuid.uuid5(uuid.NAMESPACE_DNS, hash_hex))

        except Exception as e:
            # 如果确定性生成失败，回退到随机UUID
            return self._generate_random_uuid()

    def _generate_random_uuid(self) -> str:
        """
        生成随机UUID

        Returns:
            随机UUID字符串
        """
        return str(uuid.uuid4())

    def generate_embedding_id(
        self, content_uuid: str, content_type: str, index: int = 0
    ) -> str:
        """
        为embedding生成ID

        Args:
            content_uuid: 内容UUID
            content_type: 内容类型 (description|face)
            index: 索引（用于区分同一内容的多个embedding，如多个人脸）

        Returns:
            embedding ID字符串
        """
        if content_type == "description":
            return f"{content_uuid}_desc"
        elif content_type == "face":
            return f"{content_uuid}_face_{index}"
        else:
            return f"{content_uuid}_{content_type}_{index}"

    @staticmethod
    def is_valid_uuid(uuid_string: str) -> bool:
        """
        验证UUID格式是否正确

        Args:
            uuid_string: UUID字符串

        Returns:
            是否为有效UUID
        """
        try:
            uuid.UUID(uuid_string)
            return True
        except ValueError:
            return False

    @staticmethod
    def extract_content_uuid_from_embedding_id(embedding_id: str) -> Optional[str]:
        """
        从embedding ID中提取content UUID

        Args:
            embedding_id: embedding ID字符串

        Returns:
            content UUID，如果提取失败返回None
        """
        try:
            # embedding_id格式: {uuid}_desc 或 {uuid}_face_{index}
            parts = embedding_id.split("_")
            if len(parts) >= 2:
                # 重新构建UUID部分（可能包含多个-分隔的段）
                potential_uuid_parts = []
                for part in parts:
                    potential_uuid_parts.append(part)
                    potential_uuid = "_".join(potential_uuid_parts)
                    if UUIDManager.is_valid_uuid(potential_uuid):
                        return potential_uuid
            return None
        except Exception:
            return None


# 全局UUID管理器实例
default_uuid_manager = UUIDManager(deterministic=False)


def generate_content_uuid(image_path: str, deterministic: bool = False) -> str:
    """
    便捷函数：生成内容UUID

    Args:
        image_path: 图片路径
        deterministic: 是否使用确定性生成

    Returns:
        UUID字符串
    """
    manager = UUIDManager(deterministic=deterministic)
    return manager.generate_content_uuid(image_path)


def generate_embedding_id(content_uuid: str, content_type: str, index: int = 0) -> str:
    """
    便捷函数：生成embedding ID

    Args:
        content_uuid: 内容UUID
        content_type: 内容类型
        index: 索引

    Returns:
        embedding ID字符串
    """
    return default_uuid_manager.generate_embedding_id(content_uuid, content_type, index)
