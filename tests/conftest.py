"""
pytest配置文件
"""
import os
import shutil
import tempfile
import warnings
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from dotenv import load_dotenv
from PIL import Image, ImageDraw

# 过滤SWIG相关的warnings
warnings.filterwarnings(
    "ignore", message="builtin type SwigPyPacked has no __module__ attribute"
)
warnings.filterwarnings(
    "ignore", message="builtin type SwigPyObject has no __module__ attribute"
)
warnings.filterwarnings(
    "ignore", message="builtin type swigvarlink has no __module__ attribute"
)

# 加载环境变量
load_dotenv()

# 导入测试数据管理模块
from tests.test_data import test_data


@pytest.fixture(scope="session")
def use_real_api():
    """检查是否使用真实API"""
    return os.getenv("USE_REAL_API", "false").lower() == "true"


@pytest.fixture(scope="session")
def api_key():
    """获取API密钥"""
    return os.getenv("DASHSCOPE_API_KEY", "")


@pytest.fixture
def temp_dir():
    """创建临时目录"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_image_path():
    """获取真实的示例图片路径"""
    return test_data.get_sample_landscape_path()


@pytest.fixture
def sample_portrait_path():
    """获取真实的人物图片路径"""
    return test_data.get_sample_portrait_path()


@pytest.fixture
def sample_screenshot_path():
    """获取真实的界面截图路径"""
    return test_data.get_sample_interface_path()


@pytest.fixture
def sample_images_dir():
    """获取多张真实测试图片路径"""
    return test_data.get_multiple_images(count=3)


@pytest.fixture
def sample_image_metadata():
    """示例图片元数据"""
    from schemas.data_models import ImageMetadata, ProcessingStatus

    return ImageMetadata(
        path="/test/path/image.jpg",
        is_snap=False,
        is_landscape=True,
        description="美丽的风景照",
        has_person=False,
        face_rects=[],
        unique_id="test_id_123",
        processing_status=ProcessingStatus.SUCCESS,
    )


@pytest.fixture
def sample_face_metadata():
    """包含人脸的示例图片元数据"""
    from schemas.data_models import ImageMetadata, ProcessingStatus

    return ImageMetadata(
        path="/test/path/face.jpg",
        is_snap=False,
        is_landscape=False,
        description="人物肖像照",
        has_person=True,
        face_rects=[(50, 80, 100, 100)],
        unique_id="face_id_456",
        processing_status=ProcessingStatus.SUCCESS,
    )


@pytest.fixture
def image_processor(api_key, use_real_api):
    """创建图片处理器实例"""
    if use_real_api and api_key:
        # 使用真实API
        from clients.qwen_client import QwenClient
        from config import QwenVLConfig
        from processors.image_processor import ImageProcessor

        config = QwenVLConfig(api_key=api_key)
        qwen_client = QwenClient(qwen_config=config)
        return ImageProcessor(qwen_client=qwen_client)
    else:
        # 返回None，测试中使用mock
        return None


@pytest.fixture
def faiss_store():
    """创建FAISS存储实例"""
    from vector_store import FaissStore

    return FaissStore(dimension=768)


# Mock相关的fixture（当不使用真实API时）
@pytest.fixture
def mock_qwen_api_response():
    """模拟Qwen API响应"""
    return '{"is_snap": false, "is_landscape": true, "description": "美丽的山水风景", "has_person": false, "face_rects": []}'


@pytest.fixture
def mock_qwen_api_with_face_response():
    """模拟包含人脸的Qwen API响应"""
    return '{"is_snap": false, "is_landscape": false, "description": "一个人的肖像照", "has_person": true, "face_rects": [[50, 80, 100, 100]]}'


@pytest.fixture
def sample_vectors():
    """生成示例向量数据"""
    return np.random.rand(10, 768).astype(np.float32)


@pytest.fixture
def sample_ids():
    """生成示例ID列表"""
    return [f"img_id_{i}" for i in range(10)]


# 新的客户端相关 fixtures
@pytest.fixture
def qwen_client(api_key, use_real_api):
    """创建QwenClient实例"""
    if use_real_api and api_key:
        from clients.qwen_client import QwenClient
        from config import QwenVLConfig

        config = QwenVLConfig(api_key=api_key)
        return QwenClient(qwen_config=config)
    else:
        # 返回None，测试中使用mock
        return None


@pytest.fixture
def prompt_manager():
    """创建PromptManager实例"""
    from clients.prompt_manager import PromptManager

    return PromptManager()


@pytest.fixture
def mock_qwen_client():
    """创建模拟的QwenClient"""
    from unittest.mock import Mock

    mock_client = Mock()
    mock_client.chat_with_image.return_value = '{"test": "mock_response"}'
    mock_client.chat_with_text.return_value = "Mock text response"
    mock_client.get_client_info.return_value = {
        "model": "mock-model",
        "base_url": "mock-url",
        "max_tokens": 1024,
        "temperature": 0.1,
        "timeout": 60,
    }
    return mock_client


@pytest.fixture
def test_config():
    """创建测试用的配置"""
    from config import QwenVLConfig

    return QwenVLConfig(
        api_key="test_api_key",
        model="test-model",
        temperature=0.5,
        max_tokens=512,
        timeout=30,
    )
