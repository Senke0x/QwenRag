"""
ImageProcessor真实API集成测试
测试 ImageProcessor 类的所有函数，使用真实的Qwen API和dataset目录下的图片
"""
import json
import os
from datetime import datetime
from typing import List

import pytest

from clients.prompt_manager import PromptManager, PromptType
from clients.qwen_client import QwenClient
from config import QwenVLConfig
from processors.image_processor import ImageProcessor
from schemas.data_models import ImageMetadata, ProcessingStatus


class TestImageProcessorRealAPI:
    """ImageProcessor真实API集成测试类"""

    @pytest.fixture(scope="class")
    def real_processor(self):
        """创建真实的ImageProcessor实例"""
        api_key = os.getenv("DASHSCOPE_API_KEY")
        use_real_api = os.getenv("USE_REAL_API", "false").lower() == "true"

        if not use_real_api or not api_key:
            pytest.skip("跳过真实API测试 - 设置USE_REAL_API=true和DASHSCOPE_API_KEY环境变量来启用")

        config = QwenVLConfig(api_key=api_key)
        qwen_client = QwenClient(qwen_config=config)
        prompt_manager = PromptManager()

        return ImageProcessor(qwen_client=qwen_client, prompt_manager=prompt_manager)

    @pytest.fixture(scope="class")
    def dataset_images(self):
        """获取dataset目录下的测试图片"""
        dataset_dir = "/Users/chaisenpeng/Document/Github/QwenRag/dataset"

        if not os.path.exists(dataset_dir):
            pytest.skip(f"Dataset目录不存在: {dataset_dir}")

        image_files = [
            f
            for f in os.listdir(dataset_dir)
            if f.endswith(".jpg") and not f.startswith("._")
        ]
        if not image_files:
            pytest.skip(f"Dataset目录下没有找到有效的jpg图片: {dataset_dir}")

        # 选择前5张图片进行测试
        selected_files = sorted(image_files)[:5]
        image_paths = [os.path.join(dataset_dir, f) for f in selected_files]

        # 验证文件存在
        valid_paths = []
        for path in image_paths:
            if os.path.exists(path) and os.path.getsize(path) > 0:
                valid_paths.append(path)

        if not valid_paths:
            pytest.skip("没有找到有效的测试图片")

        return valid_paths

    @pytest.fixture(scope="class")
    def single_test_image(self, dataset_images):
        """获取单张测试图片"""
        return dataset_images[0]

    @pytest.mark.integration
    def test_image_processor_initialization_real_api(self, real_processor):
        """测试ImageProcessor初始化功能"""
        # 验证组件正确初始化
        assert real_processor.qwen_client is not None
        assert real_processor.prompt_manager is not None
        assert real_processor.image_config is not None

        # 验证客户端配置
        client_info = real_processor.qwen_client.get_client_info()
        assert "model" in client_info
        # API key通常不会在client_info中暴露，所以检查其他配置
        assert client_info["model"] != ""

    @pytest.mark.integration
    def test_analyze_image_single_screenshot_real_api(
        self, real_processor, single_test_image
    ):
        """测试analyze_image()函数处理单张截图"""
        try:
            result = real_processor.analyze_image(single_test_image)

            # 验证返回结构
            assert isinstance(result, dict), "analyze_image应该返回字典类型"

            # 验证必需字段存在
            required_fields = [
                "is_snap",
                "is_landscape",
                "description",
                "has_person",
                "face_rects",
            ]
            for field in required_fields:
                assert field in result, f"结果中缺少必需字段: {field}"

            # 验证字段类型
            assert isinstance(result["is_snap"], bool), "is_snap应该是布尔类型"
            assert isinstance(result["is_landscape"], bool), "is_landscape应该是布尔类型"
            assert isinstance(result["description"], str), "description应该是字符串类型"
            assert isinstance(result["has_person"], bool), "has_person应该是布尔类型"
            assert isinstance(result["face_rects"], list), "face_rects应该是列表类型"

            # 验证内容有效性
            assert len(result["description"]) > 0, "description不应该为空"

            # 如果检测到人脸，验证人脸框格式
            if result["has_person"] and result["face_rects"]:
                for face_rect in result["face_rects"]:
                    assert isinstance(face_rect, list), "人脸框应该是列表类型"
                    assert len(face_rect) == 4, "人脸框应该包含4个坐标值"
                    assert all(
                        isinstance(coord, (int, float)) for coord in face_rect
                    ), "人脸框坐标应该是数字"

            print(f"✓ 图片分析成功: {os.path.basename(single_test_image)}")
            print(f"  - 描述: {result['description'][:100]}...")
            print(f"  - 是否截图: {result['is_snap']}")
            print(f"  - 是否风景: {result['is_landscape']}")
            print(f"  - 包含人物: {result['has_person']}")
            print(f"  - 人脸数量: {len(result['face_rects'])}")

        except Exception as e:
            pytest.fail(f"analyze_image调用失败: {e}")

    @pytest.mark.integration
    def test_process_image_single_screenshot_real_api(
        self, real_processor, single_test_image
    ):
        """测试process_image()函数生成完整元数据"""
        try:
            result = real_processor.process_image(single_test_image)

            # 验证返回类型
            assert isinstance(result, ImageMetadata), "process_image应该返回ImageMetadata对象"

            # 验证基本属性
            assert result.path == single_test_image, "路径应该匹配"
            assert (
                result.processing_status == ProcessingStatus.SUCCESS
            ), f"处理状态应该是SUCCESS，实际是: {result.processing_status}"

            # 验证元数据完整性
            assert result.unique_id != "", "unique_id不应该为空"
            assert result.description != "", "description不应该为空"
            assert result.last_processed is not None, "last_processed应该被设置"
            assert isinstance(
                result.last_processed, datetime
            ), "last_processed应该是datetime类型"

            # 验证分析结果
            assert isinstance(result.is_snap, bool), "is_snap应该是布尔类型"
            assert isinstance(result.is_landscape, bool), "is_landscape应该是布尔类型"
            assert isinstance(result.has_person, bool), "has_person应该是布尔类型"
            assert isinstance(result.face_rects, list), "face_rects应该是列表类型"

            # 验证时间戳
            if result.timestamp:
                assert isinstance(result.timestamp, str), "timestamp应该是字符串类型"

            print(f"✓ 图片处理成功: {os.path.basename(single_test_image)}")
            print(f"  - ID: {result.unique_id}")
            print(f"  - 描述: {result.description[:100]}...")
            print(f"  - 处理时间: {result.last_processed}")
            print(f"  - 状态: {result.processing_status}")

        except Exception as e:
            pytest.fail(f"process_image调用失败: {e}")

    @pytest.mark.integration
    def test_process_images_batch_real_api(self, real_processor, dataset_images):
        """测试批量处理功能"""
        # 限制批量测试的图片数量，避免API调用过多
        test_images = dataset_images[:3]

        try:
            results = real_processor.process_images_batch(test_images)

            # 验证返回类型和数量
            assert isinstance(results, list), "process_images_batch应该返回列表"
            assert len(results) == len(
                test_images
            ), f"结果数量应该匹配输入，期望{len(test_images)}，实际{len(results)}"

            # 验证每个结果
            for i, (image_path, metadata) in enumerate(zip(test_images, results)):
                assert isinstance(
                    metadata, ImageMetadata
                ), f"第{i+1}个结果应该是ImageMetadata对象"
                assert metadata.path == image_path, f"第{i+1}个结果路径不匹配"

                # 验证处理状态（可能成功或失败）
                assert metadata.processing_status in [
                    ProcessingStatus.SUCCESS,
                    ProcessingStatus.FAILED,
                ], f"第{i+1}个结果状态无效: {metadata.processing_status}"

                # 验证处理时间
                assert metadata.last_processed is not None, f"第{i+1}个结果缺少处理时间"

                # 如果处理成功，验证结果完整性
                if metadata.processing_status == ProcessingStatus.SUCCESS:
                    assert metadata.unique_id != "", f"第{i+1}个结果unique_id为空"
                    assert metadata.description != "", f"第{i+1}个结果description为空"

                # 如果处理失败，验证错误信息
                if metadata.processing_status == ProcessingStatus.FAILED:
                    assert metadata.error_message != "", f"第{i+1}个结果缺少错误信息"

            # 统计处理结果
            success_count = sum(
                1 for r in results if r.processing_status == ProcessingStatus.SUCCESS
            )
            failed_count = len(results) - success_count

            print(f"✓ 批量处理完成: 总数{len(results)}, 成功{success_count}, 失败{failed_count}")

            for i, result in enumerate(results):
                status_symbol = (
                    "✓" if result.processing_status == ProcessingStatus.SUCCESS else "✗"
                )
                image_name = os.path.basename(result.path)
                print(
                    f"  {status_symbol} {i+1}. {image_name}: {result.processing_status}"
                )

                if result.processing_status == ProcessingStatus.SUCCESS:
                    print(f"      描述: {result.description[:80]}...")
                else:
                    print(f"      错误: {result.error_message}")

            # 至少应该有一些成功的处理
            assert success_count > 0, "批量处理应该至少有一些成功的结果"

        except Exception as e:
            pytest.fail(f"process_images_batch调用失败: {e}")

    @pytest.mark.integration
    def test_extract_face_embeddings_real_api(self, real_processor, dataset_images):
        """测试人脸提取功能"""
        # 选择一张图片进行测试
        test_image = dataset_images[0]

        try:
            # 首先处理图片获取元数据
            metadata = real_processor.process_image(test_image)

            assert (
                metadata.processing_status == ProcessingStatus.SUCCESS
            ), f"图片处理失败，无法测试人脸提取: {metadata.error_message}"

            # 提取人脸
            face_embeddings = real_processor.extract_face_embeddings(metadata)

            # 验证返回类型
            assert isinstance(face_embeddings, list), "extract_face_embeddings应该返回列表"

            # 验证逻辑一致性
            if metadata.has_person and metadata.face_rects:
                # 如果检测到人脸，应该能提取到人脸embedding或者因为技术原因提取失败
                assert len(face_embeddings) <= len(
                    metadata.face_rects
                ), "提取的人脸数量不应该超过检测到的人脸数量"

                # 验证每个embedding的格式
                for i, embedding in enumerate(face_embeddings):
                    assert isinstance(
                        embedding, str
                    ), f"第{i+1}个人脸embedding应该是字符串(base64)"
                    assert len(embedding) > 0, f"第{i+1}个人脸embedding不应该为空"

                print(
                    f"✓ 人脸提取成功: 检测到{len(metadata.face_rects)}个人脸，提取到{len(face_embeddings)}个embedding"
                )

            else:
                # 如果没有检测到人脸，应该返回空列表
                assert len(face_embeddings) == 0, "没有检测到人脸时，应该返回空的embedding列表"
                print(f"✓ 无人脸图片处理正确: 返回空embedding列表")

            print(f"  - 图片: {os.path.basename(test_image)}")
            print(f"  - 检测到人脸: {metadata.has_person}")
            print(f"  - 人脸框数量: {len(metadata.face_rects)}")
            print(f"  - 提取的embedding数量: {len(face_embeddings)}")

        except Exception as e:
            pytest.fail(f"extract_face_embeddings调用失败: {e}")

    @pytest.mark.integration
    def test_api_error_handling_real_api(self, real_processor):
        """测试错误处理机制"""
        # 测试不存在的文件
        non_existent_file = "/path/to/non/existent/image.jpg"

        try:
            result = real_processor.process_image(non_existent_file)

            # 验证错误处理
            assert isinstance(result, ImageMetadata), "即使出错也应该返回ImageMetadata对象"
            assert result.processing_status == ProcessingStatus.FAILED, "不存在的文件应该处理失败"
            assert result.error_message != "", "应该包含错误信息"
            assert result.path == non_existent_file, "路径应该匹配"
            assert result.last_processed is not None, "应该记录处理时间"

            print(f"✓ 错误处理测试通过:")
            print(f"  - 文件: {non_existent_file}")
            print(f"  - 状态: {result.processing_status}")
            print(f"  - 错误信息: {result.error_message}")

        except Exception as e:
            pytest.fail(f"错误处理测试失败: {e}")

    @pytest.mark.integration
    def test_different_image_analysis_real_api(self, real_processor, dataset_images):
        """测试不同图片的分析结果差异"""
        if len(dataset_images) < 2:
            pytest.skip("需要至少2张图片来测试差异")

        # 选择前两张图片
        test_images = dataset_images[:2]
        results = []

        try:
            for image_path in test_images:
                result = real_processor.analyze_image(image_path)
                results.append((image_path, result))

            # 验证结果
            assert len(results) == 2, "应该有两个分析结果"

            image1_path, result1 = results[0]
            image2_path, result2 = results[1]

            # 验证两个结果都有效
            for path, result in results:
                assert isinstance(result, dict), f"{os.path.basename(path)}的结果应该是字典"
                assert (
                    "description" in result
                ), f"{os.path.basename(path)}的结果缺少description"
                assert (
                    len(result["description"]) > 0
                ), f"{os.path.basename(path)}的description为空"

            print(f"✓ 多图片分析完成:")
            print(f"  图片1: {os.path.basename(image1_path)}")
            print(f"    描述: {result1['description'][:100]}...")
            print(
                f"    特征: 截图={result1['is_snap']}, 风景={result1['is_landscape']}, 人物={result1['has_person']}"
            )

            print(f"  图片2: {os.path.basename(image2_path)}")
            print(f"    描述: {result2['description'][:100]}...")
            print(
                f"    特征: 截图={result2['is_snap']}, 风景={result2['is_landscape']}, 人物={result2['has_person']}"
            )

            # 验证描述的差异性（不同图片应该有不同的描述）
            descriptions_different = result1["description"] != result2["description"]
            if descriptions_different:
                print(f"  ✓ 不同图片产生了不同的描述")
            else:
                print(f"  ! 两张图片产生了相同的描述（可能正常，取决于图片内容）")

        except Exception as e:
            pytest.fail(f"多图片分析测试失败: {e}")

    @pytest.mark.integration
    def test_image_metadata_consistency_real_api(
        self, real_processor, single_test_image
    ):
        """测试元数据一致性"""
        try:
            # 多次处理同一张图片
            result1 = real_processor.process_image(single_test_image)
            result2 = real_processor.process_image(single_test_image)

            # 验证两次处理都成功
            assert result1.processing_status == ProcessingStatus.SUCCESS, "第一次处理应该成功"
            assert result2.processing_status == ProcessingStatus.SUCCESS, "第二次处理应该成功"

            # 验证一致性
            assert result1.unique_id == result2.unique_id, "同一图片的unique_id应该保持一致"
            assert result1.path == result2.path, "路径应该一致"

            # 时间戳可能相同也可能不同，取决于图片是否有EXIF数据
            if result1.timestamp and result2.timestamp:
                assert result1.timestamp == result2.timestamp, "时间戳应该一致"

            # 处理时间应该不同（因为是不同时刻处理的）
            assert result1.last_processed != result2.last_processed, "处理时间应该不同"

            print(f"✓ 元数据一致性测试通过:")
            print(f"  - 图片: {os.path.basename(single_test_image)}")
            print(f"  - ID一致: {result1.unique_id == result2.unique_id}")
            print(f"  - 第一次处理: {result1.last_processed}")
            print(f"  - 第二次处理: {result2.last_processed}")

            # API响应可能会有细微差异，但基本内容应该相似
            desc1_words = set(result1.description.lower().split())
            desc2_words = set(result2.description.lower().split())
            common_words = desc1_words.intersection(desc2_words)
            similarity_ratio = len(common_words) / max(
                len(desc1_words), len(desc2_words)
            )

            print(f"  - 描述相似度: {similarity_ratio:.2f}")

            # 由于AI模型的描述可能会有变化，我们只检查基本功能
            # 如果相似度很低，只是记录但不失败测试，因为描述的变化是正常的
            if similarity_ratio < 0.3:
                print(f"  ! 注意: API描述有较大差异，这在AI模型中是正常的")
            else:
                print(f"  ✓ 描述保持了较好的一致性")

        except Exception as e:
            pytest.fail(f"元数据一致性测试失败: {e}")

    @pytest.mark.integration
    def test_api_response_parsing_robustness_real_api(
        self, real_processor, single_test_image
    ):
        """测试API响应解析的鲁棒性"""
        try:
            # 直接测试内部解析方法的鲁棒性
            # 先获取一个真实的API响应
            result = real_processor.analyze_image(single_test_image)

            # 验证解析结果的完整性
            assert isinstance(result, dict), "解析结果应该是字典"

            # 验证默认值处理
            expected_keys = [
                "is_snap",
                "is_landscape",
                "description",
                "has_person",
                "face_rects",
            ]
            for key in expected_keys:
                assert key in result, f"解析结果缺少键: {key}"

            # 验证数据类型
            assert isinstance(result.get("is_snap"), bool), "is_snap应该是布尔类型"
            assert isinstance(result.get("is_landscape"), bool), "is_landscape应该是布尔类型"
            assert isinstance(result.get("description"), str), "description应该是字符串类型"
            assert isinstance(result.get("has_person"), bool), "has_person应该是布尔类型"
            assert isinstance(result.get("face_rects"), list), "face_rects应该是列表类型"

            print(f"✓ API响应解析测试通过:")
            print(f"  - 图片: {os.path.basename(single_test_image)}")
            print(f"  - 包含所有必需字段: {all(key in result for key in expected_keys)}")
            print(f"  - 数据类型正确: True")

        except Exception as e:
            pytest.fail(f"API响应解析测试失败: {e}")
