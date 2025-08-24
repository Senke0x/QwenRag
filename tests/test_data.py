"""
测试数据管理模块
统一管理dataset目录下的真实图片数据
"""
import os
import random
from pathlib import Path
from typing import List, Optional

from utils.image_utils import image_to_base64


class TestImageData:
    """测试图片数据管理类"""

    def __init__(self):
        # 获取项目根目录
        self.project_root = Path(__file__).parent.parent
        self.dataset_dir = self.project_root / "dataset"

        # 缓存图片路径
        self._image_paths = None
        self._categorized_images = None

    @property
    def image_paths(self) -> List[Path]:
        """获取所有图片路径"""
        if self._image_paths is None:
            self._image_paths = list(self.dataset_dir.glob("*.jpg"))
        return self._image_paths

    @property
    def categorized_images(self) -> dict:
        """按类型分类的图片"""
        if self._categorized_images is None:
            self._categorized_images = {
                # 游戏截图 - 包含人物的
                "game_with_people": [
                    "The Last of Us™ Part I_20230212234856.jpg",
                    "The Last of Us™ Part I_20230219123321.jpg",
                    "The Last of Us™ Part I_20230226110928.jpg",
                    "The Last of Us™ Part I_20230226111247.jpg",
                    "The Last of Us™ Part I_20230226144408.jpg",
                    "The Last of Us™ Part I_20230226144432.jpg",
                ],
                # 游戏截图 - 风景类
                "game_landscape": [
                    "The Last of Us™ Part I_20230219123504.jpg",
                    "The Last of Us™ Part I_20230226114722.jpg",
                    "The Last of Us™ Part I_20230226121440.jpg",
                    "The Last of Us™ Part I_20230226125222.jpg",
                    "The Last of Us™ Part I_20230226132625.jpg",
                    "The Last of Us™ Part I_20230226141454.jpg",
                ],
                # 游戏截图 - 界面/菜单类
                "game_interface": [
                    "The Last of Us™ Part I_20230226110928(1).jpg",
                    "The Last of Us™ Part I_20230226110928(2).jpg",
                    "The Last of Us™ Part I_20230226114728.jpg",
                    "The Last of Us™ Part I_20230226114817.jpg",
                    "The Last of Us™ Part I_20230226121459.jpg",
                ],
                # 游戏截图 - 其他场景
                "game_other": [
                    "The Last of Us™ Part I_20230226144436.jpg",
                    "The Last of Us™ Part I_20230226144547.jpg",
                    "The Last of Us™ Part I_20230226144613.jpg",
                    "The Last of Us™ Part I_20230226150658.jpg",
                    "The Last of Us™ Part I_20230226160149.jpg",
                    "The Last of Us™ Part I_20230301222604.jpg",
                    "The Last of Us™ Part I_20230301222642.jpg",
                ],
            }
        return self._categorized_images

    def get_sample_image_path(self) -> str:
        """获取一个示例图片路径（相对路径）"""
        if not self.image_paths:
            raise FileNotFoundError("Dataset目录下没有找到图片文件")
        return f"dataset/{self.image_paths[0].name}"

    def get_sample_portrait_path(self) -> str:
        """获取一个包含人物的示例图片路径"""
        portrait_images = self.categorized_images["game_with_people"]
        if not portrait_images:
            # 如果没有人物图片，返回第一张图片
            return self.get_sample_image_path()
        return f"dataset/{portrait_images[0]}"

    def get_sample_landscape_path(self) -> str:
        """获取一个风景类图片路径"""
        landscape_images = self.categorized_images["game_landscape"]
        if not landscape_images:
            return self.get_sample_image_path()
        return f"dataset/{landscape_images[0]}"

    def get_sample_interface_path(self) -> str:
        """获取一个界面类图片路径"""
        interface_images = self.categorized_images["game_interface"]
        if not interface_images:
            return self.get_sample_image_path()
        return f"dataset/{interface_images[0]}"

    def get_random_image_path(self, category: Optional[str] = None) -> str:
        """获取随机图片路径"""
        if category and category in self.categorized_images:
            images = self.categorized_images[category]
            return f"dataset/{random.choice(images)}"
        else:
            return f"dataset/{random.choice([p.name for p in self.image_paths])}"

    def get_multiple_images(
        self, count: int = 3, category: Optional[str] = None
    ) -> List[str]:
        """获取多张图片路径"""
        if category and category in self.categorized_images:
            images = self.categorized_images[category]
            selected = random.sample(images, min(count, len(images)))
        else:
            all_images = [p.name for p in self.image_paths]
            selected = random.sample(all_images, min(count, len(all_images)))

        return [f"dataset/{img}" for img in selected]

    def get_image_base64(self, relative_path: str) -> str:
        """获取图片的base64编码"""
        full_path = self.project_root / relative_path
        if not full_path.exists():
            raise FileNotFoundError(f"图片文件不存在: {full_path}")
        return image_to_base64(str(full_path))

    def verify_dataset(self) -> dict:
        """验证dataset目录状态"""
        info = {
            "dataset_exists": self.dataset_dir.exists(),
            "total_images": len(self.image_paths),
            "categorized_count": sum(
                len(imgs) for imgs in self.categorized_images.values()
            ),
            "categories": {
                category: len(images)
                for category, images in self.categorized_images.items()
            },
        }
        return info


# 全局测试数据实例
test_data = TestImageData()


# 便捷函数
def get_test_image() -> str:
    """获取测试图片路径"""
    return test_data.get_sample_image_path()


def get_test_portrait() -> str:
    """获取包含人物的测试图片路径"""
    return test_data.get_sample_portrait_path()


def get_test_landscape() -> str:
    """获取风景类测试图片路径"""
    return test_data.get_sample_landscape_path()


def get_test_interface() -> str:
    """获取界面类测试图片路径"""
    return test_data.get_sample_interface_path()


def get_random_test_image(category: Optional[str] = None) -> str:
    """获取随机测试图片路径"""
    return test_data.get_random_image_path(category)


def get_test_image_base64(relative_path: Optional[str] = None) -> str:
    """获取测试图片的base64编码"""
    if relative_path is None:
        relative_path = get_test_image()
    return test_data.get_image_base64(relative_path)
