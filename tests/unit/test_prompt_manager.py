"""
PromptManager 测试用例
"""
import json

import pytest

from clients.prompt_manager import PromptManager, PromptType


@pytest.mark.unit
class TestPromptManager:
    """PromptManager 测试类"""

    def test_init_default_prompts(self):
        """测试默认提示词初始化"""
        pm = PromptManager()

        # 验证默认提示词类型
        expected_types = [
            "image_analysis",
            "face_detection",
            "scene_classification",
            "text_generation",
        ]

        available_types = pm.list_prompt_types()
        for expected_type in expected_types:
            assert expected_type in available_types

        assert len(available_types) == len(expected_types)

    def test_get_prompt_image_analysis(self):
        """测试获取图像分析提示词"""
        pm = PromptManager()

        prompt = pm.get_prompt(PromptType.IMAGE_ANALYSIS)

        assert isinstance(prompt, dict)
        assert "system" in prompt
        assert "user" in prompt
        assert "图像分析助手" in prompt["system"]
        assert "JSON格式" in prompt["user"]
        assert "is_snap" in prompt["user"]
        assert "is_landscape" in prompt["user"]
        assert "description" in prompt["user"]
        assert "has_person" in prompt["user"]
        assert "face_rects" in prompt["user"]

    def test_get_prompt_face_detection(self):
        """测试获取人脸检测提示词"""
        pm = PromptManager()

        prompt = pm.get_prompt(PromptType.FACE_DETECTION)

        assert isinstance(prompt, dict)
        assert "system" in prompt
        assert "user" in prompt
        assert "人脸检测助手" in prompt["system"]
        assert "faces" in prompt["user"]
        assert "bbox" in prompt["user"]
        assert "confidence" in prompt["user"]
        assert "age_range" in prompt["user"]
        assert "gender" in prompt["user"]
        assert "expression" in prompt["user"]

    def test_get_prompt_scene_classification(self):
        """测试获取场景分类提示词"""
        pm = PromptManager()

        prompt = pm.get_prompt(PromptType.SCENE_CLASSIFICATION)

        assert isinstance(prompt, dict)
        assert "system" in prompt
        assert "user" in prompt
        assert "场景分类助手" in prompt["system"]
        assert "scene_type" in prompt["user"]
        assert "categories" in prompt["user"]
        assert "objects" in prompt["user"]
        assert "activities" in prompt["user"]
        assert "time_of_day" in prompt["user"]
        assert "weather" in prompt["user"]

    def test_get_prompt_text_generation(self):
        """测试获取文本生成提示词"""
        pm = PromptManager()

        prompt = pm.get_prompt(PromptType.TEXT_GENERATION)

        assert isinstance(prompt, dict)
        assert "system" in prompt
        assert "user" in prompt
        assert "AI助手" in prompt["system"]
        assert "{user_request}" in prompt["user"]

    def test_get_prompt_with_parameters(self):
        """测试带参数的提示词获取"""
        pm = PromptManager()

        # 测试文本生成的参数化
        prompt = pm.get_prompt(PromptType.TEXT_GENERATION, user_request="请帮我写一首关于春天的诗")

        assert "请帮我写一首关于春天的诗" in prompt["user"]
        assert "{user_request}" not in prompt["user"]

    def test_get_prompt_invalid_type(self):
        """测试获取无效类型的提示词"""
        pm = PromptManager()

        # 这应该抛出ValueError，因为枚举中没有这个值
        with pytest.raises(ValueError):
            PromptType("invalid_type")

    def test_get_system_prompt(self):
        """测试获取系统提示词"""
        pm = PromptManager()

        system_prompt = pm.get_system_prompt(PromptType.IMAGE_ANALYSIS)

        assert isinstance(system_prompt, str)
        assert "图像分析助手" in system_prompt
        assert "JSON格式" in system_prompt

    def test_get_user_prompt(self):
        """测试获取用户提示词"""
        pm = PromptManager()

        user_prompt = pm.get_user_prompt(PromptType.FACE_DETECTION)

        assert isinstance(user_prompt, str)
        assert "检测这张图片中的所有人脸" in user_prompt
        assert "JSON格式" in user_prompt

    def test_get_user_prompt_with_parameters(self):
        """测试带参数的用户提示词获取"""
        pm = PromptManager()

        user_prompt = pm.get_user_prompt(
            PromptType.TEXT_GENERATION, user_request="生成一个故事"
        )

        assert user_prompt == "生成一个故事"

    def test_add_custom_prompt(self):
        """测试添加自定义提示词"""
        pm = PromptManager()

        initial_count = len(pm.list_prompt_types())

        pm.add_prompt(
            prompt_type="custom_test",
            system_prompt="你是一个测试助手",
            user_prompt="请执行测试：{test_name}",
        )

        # 验证提示词已添加
        assert len(pm.list_prompt_types()) == initial_count + 1
        assert "custom_test" in pm.list_prompt_types()

        # 验证可以获取自定义提示词
        custom_prompts = pm._prompts["custom_test"]
        assert custom_prompts["system"] == "你是一个测试助手"
        assert custom_prompts["user"] == "请执行测试：{test_name}"

    def test_add_prompt_overwrites_existing(self):
        """测试添加提示词会覆盖现有的"""
        pm = PromptManager()

        # 添加第一个版本
        pm.add_prompt(
            prompt_type="test_prompt", system_prompt="版本1系统提示", user_prompt="版本1用户提示"
        )

        # 添加第二个版本（覆盖）
        pm.add_prompt(
            prompt_type="test_prompt", system_prompt="版本2系统提示", user_prompt="版本2用户提示"
        )

        # 验证被覆盖
        prompts = pm._prompts["test_prompt"]
        assert prompts["system"] == "版本2系统提示"
        assert prompts["user"] == "版本2用户提示"

    def test_update_prompt_system_only(self):
        """测试只更新系统提示词"""
        pm = PromptManager()

        original_prompt = pm.get_prompt(PromptType.IMAGE_ANALYSIS)
        original_user = original_prompt["user"]

        pm.update_prompt(PromptType.IMAGE_ANALYSIS, system_prompt="更新后的系统提示")

        updated_prompt = pm.get_prompt(PromptType.IMAGE_ANALYSIS)

        assert updated_prompt["system"] == "更新后的系统提示"
        assert updated_prompt["user"] == original_user  # 用户提示词应该保持不变

    def test_update_prompt_user_only(self):
        """测试只更新用户提示词"""
        pm = PromptManager()

        original_prompt = pm.get_prompt(PromptType.FACE_DETECTION)
        original_system = original_prompt["system"]

        pm.update_prompt(PromptType.FACE_DETECTION, user_prompt="更新后的用户提示")

        updated_prompt = pm.get_prompt(PromptType.FACE_DETECTION)

        assert updated_prompt["system"] == original_system  # 系统提示词应该保持不变
        assert updated_prompt["user"] == "更新后的用户提示"

    def test_update_prompt_both(self):
        """测试同时更新系统和用户提示词"""
        pm = PromptManager()

        pm.update_prompt(
            PromptType.SCENE_CLASSIFICATION,
            system_prompt="新的系统提示",
            user_prompt="新的用户提示",
        )

        updated_prompt = pm.get_prompt(PromptType.SCENE_CLASSIFICATION)

        assert updated_prompt["system"] == "新的系统提示"
        assert updated_prompt["user"] == "新的用户提示"

    def test_update_prompt_invalid_type(self):
        """测试更新无效类型的提示词"""
        pm = PromptManager()

        # 这应该抛出ValueError，因为枚举中没有这个值
        with pytest.raises(ValueError):
            fake_type = type("FakeType", (), {"value": "nonexistent"})()
            pm.update_prompt(fake_type, system_prompt="test")

    def test_list_prompt_types(self):
        """测试列出所有提示词类型"""
        pm = PromptManager()

        types = pm.list_prompt_types()

        assert isinstance(types, list)
        assert len(types) >= 4  # 至少包含4个默认类型

        # 验证包含所有默认类型
        expected_types = [
            "image_analysis",
            "face_detection",
            "scene_classification",
            "text_generation",
        ]

        for expected_type in expected_types:
            assert expected_type in types

    def test_get_all_prompts(self):
        """测试获取所有提示词"""
        pm = PromptManager()

        all_prompts = pm.get_all_prompts()

        assert isinstance(all_prompts, dict)
        assert len(all_prompts) >= 4

        # 验证每个提示词都有system和user字段
        for prompt_type, prompts in all_prompts.items():
            assert isinstance(prompts, dict)
            assert "system" in prompts
            assert "user" in prompts
            assert isinstance(prompts["system"], str)
            assert isinstance(prompts["user"], str)

    def test_get_all_prompts_returns_copy(self):
        """测试get_all_prompts返回副本而不是原始数据"""
        pm = PromptManager()

        all_prompts1 = pm.get_all_prompts()
        all_prompts2 = pm.get_all_prompts()

        # 修改返回的字典不应该影响原始数据
        all_prompts1["test_modification"] = {"system": "test", "user": "test"}

        assert "test_modification" not in all_prompts2
        assert "test_modification" not in pm.get_all_prompts()

    def test_prompt_parameter_formatting(self):
        """测试提示词参数格式化"""
        pm = PromptManager()

        # 添加带多个参数的提示词
        pm.add_prompt(
            prompt_type="multi_param_test",
            system_prompt="系统：{system_param}",
            user_prompt="用户：{user_param1} 和 {user_param2}",
        )

        prompt = pm.get_prompt(
            prompt_type="multi_param_test",
            system_param="系统参数值",
            user_param1="用户参数1",
            user_param2="用户参数2",
        )

        assert prompt["system"] == "系统：系统参数值"
        assert prompt["user"] == "用户：用户参数1 和 用户参数2"

    def test_prompt_parameter_missing(self):
        """测试缺少参数时的行为"""
        pm = PromptManager()

        pm.add_prompt(
            prompt_type="param_test",
            system_prompt="系统：{required_param}",
            user_prompt="用户提示",
        )

        # 缺少必需参数应该抛出KeyError（当提供了部分参数但缺少必需参数时）
        with pytest.raises(KeyError):
            pm.get_prompt(prompt_type="param_test", other_param="value")

    def test_prompt_parameter_extra_params(self):
        """测试额外参数的处理"""
        pm = PromptManager()

        pm.add_prompt(
            prompt_type="simple_test", system_prompt="简单系统提示", user_prompt="简单用户提示"
        )

        # 提供额外参数应该被忽略
        prompt = pm.get_prompt(prompt_type="simple_test", extra_param="这个参数会被忽略")

        assert prompt["system"] == "简单系统提示"
        assert prompt["user"] == "简单用户提示"

    def test_prompt_enum_values(self):
        """测试PromptType枚举值"""
        assert PromptType.IMAGE_ANALYSIS.value == "image_analysis"
        assert PromptType.FACE_DETECTION.value == "face_detection"
        assert PromptType.SCENE_CLASSIFICATION.value == "scene_classification"
        assert PromptType.TEXT_GENERATION.value == "text_generation"

    def test_prompt_content_quality(self):
        """测试提示词内容质量"""
        pm = PromptManager()

        # 验证图像分析提示词包含必要的字段
        image_prompt = pm.get_prompt(PromptType.IMAGE_ANALYSIS)
        required_fields = [
            "is_snap",
            "is_landscape",
            "description",
            "has_person",
            "face_rects",
        ]

        for field in required_fields:
            assert field in image_prompt["user"], f"Missing field: {field}"

        # 验证人脸检测提示词包含必要的字段
        face_prompt = pm.get_prompt(PromptType.FACE_DETECTION)
        face_fields = ["bbox", "confidence", "age_range", "gender", "expression"]

        for field in face_fields:
            assert field in face_prompt["user"], f"Missing field: {field}"

    def test_json_format_instructions(self):
        """测试JSON格式说明"""
        pm = PromptManager()

        # 所有涉及结构化输出的提示词都应该包含JSON格式说明
        structured_prompts = [
            PromptType.IMAGE_ANALYSIS,
            PromptType.FACE_DETECTION,
            PromptType.SCENE_CLASSIFICATION,
        ]

        for prompt_type in structured_prompts:
            prompt = pm.get_prompt(prompt_type)
            assert (
                "JSON" in prompt["user"] or "json" in prompt["user"]
            ), f"{prompt_type.value} should include JSON format instructions"

    def test_concurrent_access(self):
        """测试并发访问安全性"""
        import threading
        import time

        pm = PromptManager()
        results = []
        errors = []

        def access_prompts():
            try:
                for _ in range(10):
                    prompt = pm.get_prompt(PromptType.IMAGE_ANALYSIS)
                    results.append(prompt)
                    time.sleep(0.001)  # 小延迟模拟并发
            except Exception as e:
                errors.append(e)

        # 创建多个线程同时访问
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=access_prompts)
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 验证没有错误且结果一致
        assert len(errors) == 0, f"Concurrent access errors: {errors}"
        assert len(results) == 50  # 5个线程 × 10次访问

        # 所有结果应该相同
        first_result = results[0]
        for result in results[1:]:
            assert result == first_result
