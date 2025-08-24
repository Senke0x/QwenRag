"""
Prompt 管理类
"""
from enum import Enum
from typing import Any, Dict, Optional


class PromptType(Enum):
    """提示词类型枚举"""

    IMAGE_ANALYSIS = "image_analysis"
    FACE_DETECTION = "face_detection"
    SCENE_CLASSIFICATION = "scene_classification"
    TEXT_GENERATION = "text_generation"


class PromptManager:
    """提示词管理器"""

    def __init__(self):
        """初始化提示词管理器"""
        self._prompts = self._init_default_prompts()

    def _init_default_prompts(self) -> Dict[str, Dict[str, str]]:
        """初始化默认提示词"""
        return {
            PromptType.IMAGE_ANALYSIS.value: {
                "system": "你是一个专业的图像分析助手。请仔细分析图片内容，并以JSON格式返回结果。",
                "user": """请仔细分析这张图片，并以JSON格式返回以下信息：
{
    "is_snap": boolean,  // 是否是手机截图
    "is_landscape": boolean,  // 是否是风景照
    "description": "string",  // 详细的图片描述，用于语义检索
    "has_person": boolean,  // 是否有人物
    "face_rects": [[x,y,w,h], ...]  // 人脸位置框，格式为[x,y,width,height]的数组
}

注意：
1. is_snap: 判断是否为手机应用界面截图、网页截图等
2. is_landscape: 判断是否为自然风景、山水、城市景观等
3. description: 提供详细的中文描述，包括主要内容、颜色、场景等
4. has_person: 判断图片中是否包含人物（包括部分身体）
5. face_rects: 如果有人脸，提供人脸的边界框坐标

请只返回JSON格式的结果，不要包含其他文字。""",
            },
            PromptType.FACE_DETECTION.value: {
                "system": "你是一个专业的人脸检测助手。请分析图片中的人脸并返回位置信息。",
                "user": """请检测这张图片中的所有人脸，并以JSON格式返回：
{
    "faces": [
        {
            "bbox": [x, y, width, height],  // 人脸边界框
            "confidence": 0.95,  // 检测置信度
            "age_range": "25-35",  // 估计年龄范围
            "gender": "male/female/unknown",  // 性别
            "expression": "smile/neutral/sad/angry"  // 表情
        }
    ],
    "total_faces": 1
}

请只返回JSON格式的结果。""",
            },
            PromptType.SCENE_CLASSIFICATION.value: {
                "system": "你是一个专业的场景分类助手。请分析图片的场景类型。",
                "user": """请分析这张图片的场景类型，并以JSON格式返回：
{
    "scene_type": "indoor/outdoor/mixed",
    "categories": ["living_room", "kitchen", ...],  // 具体场景分类
    "objects": ["sofa", "table", "person", ...],  // 主要物体
    "activities": ["cooking", "reading", ...],  // 可能的活动
    "time_of_day": "morning/afternoon/evening/night/unknown",
    "weather": "sunny/cloudy/rainy/snowy/unknown"
}

请只返回JSON格式的结果。""",
            },
            PromptType.TEXT_GENERATION.value: {
                "system": "你是一个有用的AI助手。请根据用户的要求生成相应的文本内容。",
                "user": "{user_request}",  # 占位符，会被实际请求替换
            },
        }

    def get_prompt(self, prompt_type, **kwargs) -> Dict[str, str]:
        """
        获取指定类型的提示词

        Args:
            prompt_type: 提示词类型（PromptType 枚举或字符串）
            **kwargs: 用于格式化提示词的参数

        Returns:
            包含 system 和 user 提示词的字典
        """
        # 处理不同类型的 prompt_type
        if isinstance(prompt_type, PromptType):
            type_key = prompt_type.value
        elif isinstance(prompt_type, str):
            type_key = prompt_type
        else:
            raise ValueError(
                f"prompt_type 必须是 PromptType 枚举或字符串，获得: {type(prompt_type)}"
            )

        if type_key not in self._prompts:
            raise ValueError(f"未找到提示词类型: {type_key}")

        prompt_dict = self._prompts[type_key].copy()

        # 格式化提示词
        for key, value in prompt_dict.items():
            if isinstance(value, str) and kwargs:
                try:
                    # 只有当有参数时才进行格式化
                    prompt_dict[key] = value.format(**kwargs)
                except KeyError as e:
                    # 如果缺少必需的参数，重新抛出更清晰的错误
                    raise KeyError(f"缺少必需的参数 {e} 用于格式化 {key} 提示词") from e

        return prompt_dict

    def get_system_prompt(self, prompt_type: PromptType) -> str:
        """获取系统提示词"""
        prompt_dict = self.get_prompt(prompt_type)
        return prompt_dict.get("system", "")

    def get_user_prompt(self, prompt_type: PromptType, **kwargs) -> str:
        """获取用户提示词"""
        prompt_dict = self.get_prompt(prompt_type, **kwargs)
        return prompt_dict.get("user", "")

    def add_prompt(self, prompt_type: str, system_prompt: str, user_prompt: str):
        """
        添加自定义提示词

        Args:
            prompt_type: 提示词类型名称
            system_prompt: 系统提示词
            user_prompt: 用户提示词
        """
        self._prompts[prompt_type] = {"system": system_prompt, "user": user_prompt}

    def update_prompt(
        self,
        prompt_type: PromptType,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
    ):
        """
        更新现有提示词

        Args:
            prompt_type: 提示词类型
            system_prompt: 新的系统提示词
            user_prompt: 新的用户提示词
        """
        if prompt_type.value not in self._prompts:
            raise ValueError(f"未找到提示词类型: {prompt_type.value}")

        if system_prompt is not None:
            self._prompts[prompt_type.value]["system"] = system_prompt

        if user_prompt is not None:
            self._prompts[prompt_type.value]["user"] = user_prompt

    def list_prompt_types(self) -> list:
        """列出所有可用的提示词类型"""
        return list(self._prompts.keys())

    def get_all_prompts(self) -> Dict[str, Dict[str, str]]:
        """获取所有提示词"""
        return self._prompts.copy()


# 全局提示词管理器实例
prompt_manager = PromptManager()
