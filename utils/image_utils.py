"""
图像处理相关的辅助函数
"""
import base64
import io
import hashlib
from pathlib import Path
from typing import Tuple, Optional, List
from PIL import Image, ImageOps, ExifTags
import logging

from config import ImageProcessorConfig

logger = logging.getLogger(__name__)


def get_supported_image_extensions() -> List[str]:
    """
    获取支持的图片格式扩展名列表
    
    Returns:
        支持的图片格式扩展名列表
    """
    return ['jpg', 'jpeg', 'png', 'bmp', 'gif', 'webp', 'tiff']


def is_supported_image_format(file_path: str, config: ImageProcessorConfig) -> bool:
    """
    检查文件是否为支持的图片格式
    
    Args:
        file_path: 文件路径
        config: 图片处理配置
    
    Returns:
        是否为支持的格式
    """
    file_path = Path(file_path)
    return file_path.suffix.lower() in config.supported_formats


def validate_image_file(file_path: str, config: ImageProcessorConfig) -> Tuple[bool, str]:
    """
    验证图片文件是否有效
    
    Args:
        file_path: 图片文件路径
        config: 图片处理配置
    
    Returns:
        (是否有效, 错误消息)
    """
    try:
        file_path = Path(file_path)
        
        # 检查文件是否存在
        if not file_path.exists():
            return False, f"文件不存在: {file_path}"
        
        # 检查文件大小
        file_size = file_path.stat().st_size
        if file_size > config.max_image_size:
            return False, f"文件大小超过限制: {file_size} > {config.max_image_size}"
        
        # 检查文件格式
        if not is_supported_image_format(str(file_path), config):
            return False, f"不支持的文件格式: {file_path.suffix}"
        
        # 尝试打开图片验证完整性
        with Image.open(file_path) as img:
            img.verify()
        
        # 重新打开获取尺寸信息
        with Image.open(file_path) as img:
            width, height = img.size
            max_width, max_height = config.max_resolution
            
            if width > max_width or height > max_height:
                return False, f"图片分辨率超过限制: {width}x{height} > {max_width}x{max_height}"
        
        return True, ""
    
    except Exception as e:
        return False, f"图片验证失败: {str(e)}"


def image_to_base64(image_path: str, max_size: Optional[Tuple[int, int]] = None) -> str:
    """
    将图片转换为base64编码
    
    Args:
        image_path: 图片路径
        max_size: 最大尺寸 (width, height)，如果提供则会等比例缩放
    
    Returns:
        base64编码的图片数据
    """
    try:
        with Image.open(image_path) as img:
            # 处理EXIF旋转信息
            img = ImageOps.exif_transpose(img)
            
            # 转换为RGB模式（处理透明通道）
            if img.mode in ('RGBA', 'LA', 'P'):
                # 创建白色背景
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 等比例缩放
            if max_size:
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # 转换为base64
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=85, optimize=True)
            img_data = buffer.getvalue()
            
            return base64.b64encode(img_data).decode('utf-8')
    
    except Exception as e:
        logger.error(f"图片转base64失败: {image_path}, 错误: {e}")
        raise


def crop_face_from_image(
    image_path: str, 
    face_rect: Tuple[int, int, int, int],
    padding: float = 0.2
) -> str:
    """
    从图片中裁剪人脸区域
    
    Args:
        image_path: 原图路径
        face_rect: 人脸框 (x, y, width, height)
        padding: 扩展比例，增加周围区域
    
    Returns:
        裁剪人脸的base64编码
    """
    try:
        with Image.open(image_path) as img:
            # 处理EXIF旋转
            img = ImageOps.exif_transpose(img)
            
            x, y, w, h = face_rect
            img_width, img_height = img.size
            
            # 计算扩展后的区域
            padding_w = int(w * padding)
            padding_h = int(h * padding)
            
            # 扩展边界框
            x1 = max(0, x - padding_w)
            y1 = max(0, y - padding_h)
            x2 = min(img_width, x + w + padding_w)
            y2 = min(img_height, y + h + padding_h)
            
            # 裁剪人脸区域
            face_img = img.crop((x1, y1, x2, y2))
            
            # 转换为RGB
            if face_img.mode != 'RGB':
                if face_img.mode in ('RGBA', 'LA', 'P'):
                    background = Image.new('RGB', face_img.size, (255, 255, 255))
                    if face_img.mode == 'P':
                        face_img = face_img.convert('RGBA')
                    background.paste(face_img, mask=face_img.split()[-1] if face_img.mode in ('RGBA', 'LA') else None)
                    face_img = background
                else:
                    face_img = face_img.convert('RGB')
            
            # 转换为base64
            buffer = io.BytesIO()
            face_img.save(buffer, format='JPEG', quality=85)
            img_data = buffer.getvalue()
            
            return base64.b64encode(img_data).decode('utf-8')
    
    except Exception as e:
        logger.error(f"人脸裁剪失败: {image_path}, 人脸框: {face_rect}, 错误: {e}")
        raise


def extract_image_timestamp(image_path: str) -> Optional[str]:
    """
    从图片EXIF数据中提取时间戳
    
    Args:
        image_path: 图片路径
    
    Returns:
        时间戳字符串，如果提取失败则返回None
    """
    try:
        with Image.open(image_path) as img:
            exif_data = img._getexif()
            
            if exif_data is not None:
                for tag, value in exif_data.items():
                    tag_name = ExifTags.TAGS.get(tag, tag)
                    
                    # 寻找时间相关的标签
                    if tag_name in ['DateTime', 'DateTimeOriginal', 'DateTimeDigitized']:
                        return str(value)
            
            # 如果没有EXIF时间，使用文件修改时间
            file_path = Path(image_path)
            return file_path.stat().st_mtime
    
    except Exception as e:
        logger.warning(f"提取图片时间戳失败: {image_path}, 错误: {e}")
        return None


def generate_image_id(image_path: str) -> str:
    """
    为图片生成唯一ID
    
    Args:
        image_path: 图片路径
    
    Returns:
        唯一ID字符串
    """
    try:
        file_path = Path(image_path)
        
        # 使用文件路径和修改时间生成ID
        id_string = f"{file_path.absolute()}_{file_path.stat().st_mtime}"
        
        # 计算MD5哈希
        return hashlib.md5(id_string.encode('utf-8')).hexdigest()
    
    except Exception as e:
        logger.error(f"生成图片ID失败: {image_path}, 错误: {e}")
        # 降级方案：仅使用文件路径
        return hashlib.md5(str(image_path).encode('utf-8')).hexdigest()


def get_image_info(image_path: str) -> dict:
    """
    获取图片基本信息
    
    Args:
        image_path: 图片路径
    
    Returns:
        包含图片信息的字典
    """
    try:
        file_path = Path(image_path)
        
        with Image.open(image_path) as img:
            # 处理EXIF旋转后的尺寸
            img = ImageOps.exif_transpose(img)
            width, height = img.size
            
            return {
                'path': str(file_path.absolute()),
                'filename': file_path.name,
                'size': file_path.stat().st_size,
                'width': width,
                'height': height,
                'format': img.format,
                'mode': img.mode,
                'timestamp': extract_image_timestamp(image_path),
                'unique_id': generate_image_id(image_path)
            }
    
    except Exception as e:
        logger.error(f"获取图片信息失败: {image_path}, 错误: {e}")
        raise


def find_images_in_directory(directory: str, config: ImageProcessorConfig) -> List[str]:
    """
    在目录中查找所有支持的图片文件
    
    Args:
        directory: 目录路径
        config: 图片处理配置
    
    Returns:
        图片文件路径列表
    """
    image_files = []
    directory_path = Path(directory)
    
    if not directory_path.exists():
        logger.warning(f"目录不存在: {directory}")
        return image_files
    
    try:
        for file_path in directory_path.rglob('*'):
            if file_path.is_file() and is_supported_image_format(str(file_path), config):
                image_files.append(str(file_path))
        
        logger.info(f"在目录 {directory} 中找到 {len(image_files)} 个图片文件")
        return sorted(image_files)
    
    except Exception as e:
        logger.error(f"搜索图片文件失败: {directory}, 错误: {e}")
        return image_files