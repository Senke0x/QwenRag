"""
pytest配置文件
"""
import pytest
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
from PIL import Image, ImageDraw
import numpy as np
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


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
def sample_image_path(temp_dir):
    """创建示例图片文件"""
    # 创建一个更真实的测试图片
    img = Image.new('RGB', (300, 200), color='skyblue')
    draw = ImageDraw.Draw(img)
    
    # 绘制简单的风景：天空、山、树
    # 绘制山峰
    draw.polygon([(50, 150), (150, 80), (250, 150)], fill='gray')
    # 绘制树
    draw.rectangle([100, 120, 110, 150], fill='brown')  # 树干
    draw.ellipse([90, 100, 120, 130], fill='green')     # 树冠
    
    image_path = os.path.join(temp_dir, "test_landscape.jpg")
    img.save(image_path, quality=90)
    return image_path


@pytest.fixture
def sample_portrait_path(temp_dir):
    """创建人物肖像测试图片"""
    img = Image.new('RGB', (200, 300), color='lightgray')
    draw = ImageDraw.Draw(img)
    
    # 绘制简单的人脸
    # 脸部轮廓
    draw.ellipse([50, 80, 150, 180], fill='peachpuff')
    # 眼睛
    draw.ellipse([70, 110, 85, 125], fill='white')
    draw.ellipse([115, 110, 130, 125], fill='white')
    draw.ellipse([75, 115, 80, 120], fill='black')
    draw.ellipse([120, 115, 125, 120], fill='black')
    # 鼻子
    draw.ellipse([95, 130, 105, 140], fill='pink')
    # 嘴巴
    draw.arc([85, 145, 115, 160], 0, 180, fill='red', width=2)
    
    image_path = os.path.join(temp_dir, "test_portrait.jpg")
    img.save(image_path, quality=90)
    return image_path


@pytest.fixture
def sample_screenshot_path(temp_dir):
    """创建手机截图测试图片"""
    img = Image.new('RGB', (375, 667), color='white')  # iPhone尺寸
    draw = ImageDraw.Draw(img)
    
    # 绘制状态栏
    draw.rectangle([0, 0, 375, 40], fill='black')
    draw.text([10, 15], "9:41", fill='white')
    draw.text([320, 15], "100%", fill='white')
    
    # 绘制应用界面
    draw.rectangle([20, 60, 355, 100], fill='blue')
    draw.text([30, 75], "App Title", fill='white')
    
    # 绘制按钮
    for i in range(3):
        y = 120 + i * 60
        draw.rectangle([50, y, 325, y + 40], fill='lightblue')
        draw.text([60, y + 15], f"Button {i+1}", fill='black')
    
    image_path = os.path.join(temp_dir, "test_screenshot.png")
    img.save(image_path)
    return image_path


@pytest.fixture
def sample_images_dir(temp_dir, sample_image_path, sample_portrait_path, sample_screenshot_path):
    """创建包含多个示例图片的目录"""
    images_dir = os.path.join(temp_dir, "images")
    os.makedirs(images_dir)
    
    # 复制测试图片到目录
    created_paths = []
    
    # 复制风景照
    landscape_path = os.path.join(images_dir, "landscape.jpg")
    shutil.copy2(sample_image_path, landscape_path)
    created_paths.append(landscape_path)
    
    # 复制人物照
    portrait_path = os.path.join(images_dir, "portrait.jpg")
    shutil.copy2(sample_portrait_path, portrait_path)
    created_paths.append(portrait_path)
    
    # 复制截图
    screenshot_path = os.path.join(images_dir, "screenshot.png")
    shutil.copy2(sample_screenshot_path, screenshot_path)
    created_paths.append(screenshot_path)
    
    # 创建损坏文件
    corrupted_path = os.path.join(images_dir, "corrupted.txt")
    with open(corrupted_path, 'w') as f:
        f.write("not an image")
    created_paths.append(corrupted_path)
    
    return images_dir, created_paths


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
        processing_status=ProcessingStatus.SUCCESS
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
        processing_status=ProcessingStatus.SUCCESS
    )


@pytest.fixture
def image_processor(api_key, use_real_api):
    """创建图片处理器实例"""
    if use_real_api and api_key:
        # 使用真实API
        from processors.image_processor import ImageProcessor
        from clients.qwen_client import QwenClient
        from config import QwenVLConfig
        
        config = QwenVLConfig(api_key=api_key)
        qwen_client = QwenClient(qwen_config=config)
        return ImageProcessor(qwen_client=qwen_client)
    else:
        # 返回None，测试中使用mock
        return None


@pytest.fixture
def faiss_store():
    """创建FAISS存储实例"""
    from vector_store.faiss_store import FaissStore
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
        "timeout": 60
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
        timeout=30
    )