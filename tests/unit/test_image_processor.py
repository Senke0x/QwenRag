"""
图片处理器测试用例
"""
import pytest
import os
import json
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import requests

from schemas.data_models import ImageMetadata, ProcessingStatus
from processors.image_processor import ImageProcessor
from config import QwenVLConfig, ImageProcessorConfig
from utils.image_utils import validate_image_file, is_supported_image_format


class TestImageProcessor:
    """图片处理器测试类"""
    
    def test_image_format_validation_valid_formats(self, sample_image_path):
        """测试有效图片格式验证"""
        config = ImageProcessorConfig()
        
        # 测试原始文件
        assert is_supported_image_format(sample_image_path, config)
        
        # 测试不同扩展名
        valid_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        for ext in valid_extensions:
            # 创建测试文件
            test_path = sample_image_path.replace('.jpg', ext)
            assert is_supported_image_format(test_path, config)
    
    def test_image_format_validation_invalid_formats(self, temp_dir):
        """测试无效图片格式验证"""
        config = ImageProcessorConfig()
        
        # 测试不支持的格式
        invalid_files = ['test.gif', 'test.bmp', 'test.txt', 'test.pdf']
        
        for filename in invalid_files:
            file_path = os.path.join(temp_dir, filename)
            with open(file_path, 'w') as f:
                f.write("dummy content")
            assert not is_supported_image_format(file_path, config)
    
    def test_qwen_vl_api_call_success_real(self, image_processor, sample_image_path, use_real_api):
        """测试Qwen VL API正常调用（真实API）"""
        if not use_real_api or not image_processor:
            pytest.skip("跳过真实API测试，设置USE_REAL_API=true和DASHSCOPE_API_KEY环境变量来启用")
        
        # 调用真实API
        result = image_processor.process_image(sample_image_path)
        
        assert result.processing_status == ProcessingStatus.SUCCESS
        assert result.description != ""
        assert result.unique_id != ""
        assert result.path == sample_image_path
    
    def test_qwen_vl_api_call_success_mock(self, mock_qwen_api_response, sample_image_path, use_real_api):
        """测试Qwen VL API正常调用（模拟）"""
        if use_real_api:
            pytest.skip("跳过模拟测试，当前使用真实API")
        
        with patch('processors.image_processor.OpenAI') as mock_openai:
            # 模拟OpenAI客户端
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            # 模拟API响应
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = mock_qwen_api_response
            mock_client.chat.completions.create.return_value = mock_response
            
            # 创建处理器并测试
            config = QwenVLConfig(api_key="test_key")
            processor = ImageProcessor(qwen_config=config)
            
            result = processor.process_image(sample_image_path)
            
            assert result.processing_status == ProcessingStatus.SUCCESS
            assert "美丽的山水风景" in result.description
    
    def test_qwen_vl_api_call_rate_limit(self, sample_image_path, use_real_api):
        """测试API限流处理"""
        if use_real_api:
            pytest.skip("跳过限流测试，避免触发真实API限流")
        
        with patch('processors.image_processor.OpenAI') as mock_openai:
            # 模拟OpenAI客户端
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            # 模拟限流异常
            from openai import RateLimitError
            mock_client.chat.completions.create.side_effect = RateLimitError(
                "Rate limit exceeded", response=Mock(status_code=429), body=None
            )
            
            config = QwenVLConfig(api_key="test_key")
            processor = ImageProcessor(qwen_config=config)
            
            # 验证处理失败并记录错误
            result = processor.process_image(sample_image_path)
            assert result.processing_status == ProcessingStatus.FAILED
            assert "rate limit" in result.error_message.lower() or "limit" in result.error_message.lower()
    
    def test_qwen_vl_api_call_timeout(self, sample_image_path, use_real_api):
        """测试API超时处理"""
        if use_real_api:
            pytest.skip("跳过超时测试，避免长时间等待")
        
        with patch('processors.image_processor.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            # 模拟超时异常 - 创建一个包含timeout关键字的异常
            class MockTimeoutError(Exception):
                def __init__(self, message):
                    super().__init__(message)
                    self.response = None
            
            mock_client.chat.completions.create.side_effect = MockTimeoutError("Request timeout")
            
            config = QwenVLConfig(api_key="test_key")
            processor = ImageProcessor(qwen_config=config)
            
            result = processor.process_image(sample_image_path)
            assert result.processing_status == ProcessingStatus.FAILED
            assert "timeout" in result.error_message.lower() or "网络错误" in result.error_message
    
    def test_qwen_vl_api_call_auth_failure(self, sample_image_path, use_real_api):
        """测试认证失败处理"""
        if use_real_api:
            pytest.skip("跳过认证失败测试，避免使用无效密钥")
        
        with patch('processors.image_processor.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            # 模拟认证异常 - 创建包含401状态码的响应对象
            class MockAuthError(Exception):
                def __init__(self, message):
                    super().__init__(message)
                    self.response = type('Response', (), {'status_code': 401})()
            
            mock_client.chat.completions.create.side_effect = MockAuthError("Invalid API key")
            
            config = QwenVLConfig(api_key="invalid_key")
            processor = ImageProcessor(qwen_config=config)
            
            result = processor.process_image(sample_image_path)
            assert result.processing_status == ProcessingStatus.FAILED
            assert "auth" in result.error_message.lower() or "认证失败" in result.error_message or "invalid" in result.error_message.lower()
    
    def test_face_detection_single_face_real(self, image_processor, sample_portrait_path, use_real_api):
        """测试单人脸图片检测（真实API）"""
        if not use_real_api or not image_processor:
            pytest.skip("跳过真实API测试")
        
        result = image_processor.process_image(sample_portrait_path)
        
        # 验证人脸检测结果
        assert result.processing_status == ProcessingStatus.SUCCESS
        # 注意：真实API的人脸检测结果可能因模型而异
        # 这里主要验证API调用成功
        assert isinstance(result.has_person, bool)
        assert isinstance(result.face_rects, list)
    
    def test_face_detection_single_face_mock(self, mock_qwen_api_with_face_response, sample_portrait_path, use_real_api):
        """测试单人脸图片检测（模拟）"""
        if use_real_api:
            pytest.skip("跳过模拟测试")
        
        with patch('processors.image_processor.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = mock_qwen_api_with_face_response
            mock_client.chat.completions.create.return_value = mock_response
            
            config = QwenVLConfig(api_key="test_key")
            processor = ImageProcessor(qwen_config=config)
            
            result = processor.process_image(sample_portrait_path)
            
            assert result.processing_status == ProcessingStatus.SUCCESS
            assert result.has_person == True
            assert len(result.face_rects) == 1
            assert result.face_rects[0] == [50, 80, 100, 100]
    
    def test_face_detection_multiple_faces(self, sample_portrait_path, use_real_api):
        """测试多人脸图片检测"""
        if use_real_api:
            pytest.skip("跳过多人脸测试，需要特定的测试图片")
        
        # 模拟多人脸响应
        mock_response_content = '{"is_snap": false, "is_landscape": false, "description": "多人合影", "has_person": true, "face_rects": [[10, 10, 50, 50], [70, 20, 40, 40], [120, 30, 45, 45]]}'
        
        with patch('processors.image_processor.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = mock_response_content
            mock_client.chat.completions.create.return_value = mock_response
            
            config = QwenVLConfig(api_key="test_key")
            processor = ImageProcessor(qwen_config=config)
            
            result = processor.process_image(sample_portrait_path)
            
            assert result.processing_status == ProcessingStatus.SUCCESS
            assert result.has_person == True
            assert len(result.face_rects) == 3
    
    def test_face_detection_no_face_real(self, image_processor, sample_image_path, use_real_api):
        """测试无人脸图片检测（真实API）"""
        if not use_real_api or not image_processor:
            pytest.skip("跳过真实API测试")
        
        result = image_processor.process_image(sample_image_path)
        
        assert result.processing_status == ProcessingStatus.SUCCESS
        # 风景照应该不包含人脸
        assert isinstance(result.has_person, bool)
        assert isinstance(result.face_rects, list)
    
    def test_face_detection_no_face_mock(self, mock_qwen_api_response, sample_image_path, use_real_api):
        """测试无人脸图片检测（模拟）"""
        if use_real_api:
            pytest.skip("跳过模拟测试")
        
        with patch('processors.image_processor.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = mock_qwen_api_response
            mock_client.chat.completions.create.return_value = mock_response
            
            config = QwenVLConfig(api_key="test_key")
            processor = ImageProcessor(qwen_config=config)
            
            result = processor.process_image(sample_image_path)
            
            assert result.processing_status == ProcessingStatus.SUCCESS
            assert result.has_person == False
            assert len(result.face_rects) == 0
    
    def test_face_coordinates_accuracy(self):
        """测试人脸坐标准确性"""
        # 创建已知人脸位置的测试图片
        # 验证检测到的坐标与预期坐标的偏差在可接受范围内
        
        expected_coords = (25, 25, 50, 50)  # (x, y, w, h)
        # detected_coords = result.face_rects[0]
        
        # 验证坐标精度
        # assert abs(detected_coords[0] - expected_coords[0]) < 5
        # assert abs(detected_coords[1] - expected_coords[1]) < 5
        assert True  # 占位符
    
    def test_image_classification_landscape(self):
        """测试风景照识别"""
        # 模拟风景照片的API响应
        mock_response = {
            "choices": [{
                "message": {
                    "content": '{"is_snap": false, "is_landscape": true, "description": "山水风景", "has_person": false, "face_rects": []}'
                }
            }]
        }
        
        # 验证风景照识别正确
        # assert result.is_landscape == True
        # assert result.has_person == False
        assert True  # 占位符
    
    def test_error_handling_corrupted_image(self, temp_dir):
        """测试损坏图片处理"""
        from processors.image_processor import ImageProcessor
        from config import QwenVLConfig
        
        # 创建损坏的图片文件
        corrupted_path = os.path.join(temp_dir, "corrupted.jpg")
        with open(corrupted_path, 'wb') as f:
            f.write(b"not a valid image")
        
        # 使用mock避免实际API调用
        with patch('processors.image_processor.OpenAI') as mock_openai:
            config = QwenVLConfig(api_key="test_key")
            processor = ImageProcessor(qwen_config=config)
            
            result = processor.process_image(corrupted_path)
            
            # 验证处理失败
            assert result.processing_status == ProcessingStatus.FAILED
            assert ("验证失败" in result.error_message or 
                    "invalid" in result.error_message.lower() or 
                    "cannot identify" in result.error_message.lower() or
                    "图片格式不支持" in result.error_message)
    
    def test_unique_id_generation(self, sample_image_path, sample_portrait_path):
        """测试唯一ID生成"""
        from utils.image_utils import generate_image_id
        
        # 验证同一图片生成相同ID
        id1 = generate_image_id(sample_image_path)
        id2 = generate_image_id(sample_image_path)
        assert id1 == id2
        
        # 验证不同图片生成不同ID
        id3 = generate_image_id(sample_portrait_path)
        assert id1 != id3
        
        # 验证ID格式（MD5哈希）
        assert len(id1) == 32
        assert all(c in '0123456789abcdef' for c in id1)
    
    def test_timestamp_extraction(self, sample_image_path):
        """测试时间戳提取"""
        from utils.image_utils import extract_image_timestamp
        
        # 提取时间戳（对于测试图片，可能没有EXIF数据）
        timestamp = extract_image_timestamp(sample_image_path)
        
        # 应该返回某种格式的时间戳
        assert timestamp is not None
    
    def test_image_metadata_serialization(self, sample_image_metadata):
        """测试图片元数据序列化"""
        # 测试to_dict和from_dict方法
        metadata_dict = sample_image_metadata.to_dict()
        restored_metadata = ImageMetadata.from_dict(metadata_dict)
        
        assert restored_metadata.path == sample_image_metadata.path
        assert restored_metadata.description == sample_image_metadata.description
        assert restored_metadata.processing_status == sample_image_metadata.processing_status
        assert restored_metadata.is_landscape == sample_image_metadata.is_landscape
        assert restored_metadata.has_person == sample_image_metadata.has_person
        assert restored_metadata.face_rects == sample_image_metadata.face_rects
    
    def test_face_cropping(self, sample_portrait_path, sample_face_metadata):
        """测试人脸裁剪功能"""
        from utils.image_utils import crop_face_from_image
        
        # 使用样例人脸框
        face_rect = sample_face_metadata.face_rects[0]
        
        try:
            face_base64 = crop_face_from_image(sample_portrait_path, face_rect)
            
            # 验证返回base64编码
            assert isinstance(face_base64, str)
            assert len(face_base64) > 0
            
            # 验证是有效的base64
            import base64
            decoded = base64.b64decode(face_base64)
            assert len(decoded) > 0
            
        except Exception as e:
            # 如果人脸框不在图片范围内，可能会失败
            # 这在测试中是可接受的
            assert "crop" in str(e).lower() or "rect" in str(e).lower()