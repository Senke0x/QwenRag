#!/usr/bin/env python3
"""
快速测试图片embedding功能
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from clients.qwen_client import QwenClient
from tests.test_data import get_test_image_base64

def main():
    try:
        client = QwenClient()
        print("✅ QwenClient初始化成功")
        
        image_base64 = get_test_image_base64()
        print("📷 获取测试图片base64数据")
        
        # 测试图片embedding
        result = client.get_image_embedding(image_base64)
        print(f"✅ 图片embedding成功: 维度={len(result['embedding'])}, 模型={result['model']}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")

if __name__ == "__main__":
    main()