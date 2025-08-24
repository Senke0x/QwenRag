#!/usr/bin/env python3
"""
QwenRag API演示测试脚本
测试人脸识别和Web API功能的基本使用
"""
import asyncio
import json
import time
from pathlib import Path

import requests

# API服务地址
API_BASE_URL = "http://localhost:8000"


def test_api_health():
    """测试API健康检查"""
    print("🏥 测试API健康检查...")

    try:
        response = requests.get(f"{API_BASE_URL}/health")
        print(f"响应状态: {response.status_code}")
        print(f"响应内容: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ 健康检查失败: {e}")
        return False


def test_search_health():
    """测试搜索服务健康检查"""
    print("\n🔍 测试搜索服务健康检查...")

    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/search/health")
        print(f"响应状态: {response.status_code}")
        print(f"响应内容: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ 搜索服务健康检查失败: {e}")
        return False


def test_face_health():
    """测试人脸识别服务健康检查"""
    print("\n👤 测试人脸识别服务健康检查...")

    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/faces/health")
        print(f"响应状态: {response.status_code}")
        print(f"响应内容: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ 人脸识别服务健康检查失败: {e}")
        return False


def test_index_status():
    """测试索引状态查询"""
    print("\n📊 测试索引状态查询...")

    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/index/status")
        print(f"响应状态: {response.status_code}")
        result = response.json()
        print(f"响应内容: {json.dumps(result, indent=2, ensure_ascii=False)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ 索引状态查询失败: {e}")
        return False


def test_text_search():
    """测试文本搜索"""
    print("\n🔤 测试文本搜索...")

    try:
        search_data = {"query": "美丽的风景", "limit": 5, "similarity_threshold": 0.5}

        response = requests.post(f"{API_BASE_URL}/api/v1/search/text", json=search_data)

        print(f"响应状态: {response.status_code}")
        result = response.json()
        print(f"搜索结果: 找到{result.get('total_found', 0)}个结果")

        if result.get("results"):
            print("前3个结果:")
            for i, item in enumerate(result["results"][:3]):
                print(
                    f"  {i+1}. {item['image_info']['filename']} (相似度: {item['similarity_score']:.3f})"
                )

        return response.status_code == 200

    except Exception as e:
        print(f"❌ 文本搜索失败: {e}")
        return False


def test_image_upload_and_search():
    """测试图片上传和搜索"""
    print("\n🖼️ 测试图片上传和搜索...")

    # 查找测试图片
    test_image = None
    for img_path in Path("dataset").glob("*.jpg"):
        if img_path.exists():
            test_image = img_path
            break

    if not test_image:
        print("❌ 找不到测试图片，跳过图片搜索测试")
        return False

    try:
        print(f"使用测试图片: {test_image}")

        with open(test_image, "rb") as f:
            files = {"file": (test_image.name, f, "image/jpeg")}
            params = {
                "limit": 5,
                "similarity_threshold": 0.3,
                "include_metadata": True,
                "search_faces": False,
            }

            response = requests.post(
                f"{API_BASE_URL}/api/v1/search/image", files=files, params=params
            )

        print(f"响应状态: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"搜索结果: 找到{result.get('total_found', 0)}个相似图片")
            return True
        else:
            print(f"搜索失败: {response.text}")
            return False

    except Exception as e:
        print(f"❌ 图片搜索失败: {e}")
        return False


def test_face_detection():
    """测试人脸检测"""
    print("\n👁️ 测试人脸检测...")

    # 查找测试图片
    test_image = None
    for img_path in Path("dataset").glob("*.jpg"):
        if img_path.exists():
            test_image = img_path
            break

    if not test_image:
        print("❌ 找不到测试图片，跳过人脸检测测试")
        return False

    try:
        print(f"使用测试图片: {test_image}")

        with open(test_image, "rb") as f:
            files = {"file": (test_image.name, f, "image/jpeg")}

            response = requests.post(f"{API_BASE_URL}/api/v1/faces/detect", files=files)

        print(f"响应状态: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            face_count = result.get("face_count", 0)
            print(f"检测结果: 发现{face_count}个人脸")

            if face_count > 0:
                print("人脸信息:")
                for i, face in enumerate(result.get("faces", [])):
                    bbox = face["bounding_box"]
                    print(
                        f"  人脸{i+1}: 位置({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}) 置信度:{face['confidence']:.3f}"
                    )

            return True
        else:
            print(f"人脸检测失败: {response.text}")
            return False

    except Exception as e:
        print(f"❌ 人脸检测失败: {e}")
        return False


def test_index_build():
    """测试索引构建（谨慎使用）"""
    print("\n🏗️ 测试索引构建...")

    # 检查是否有数据集目录
    dataset_dir = Path("dataset")
    if not dataset_dir.exists():
        print("❌ 找不到dataset目录，跳过索引构建测试")
        return False

    try:
        build_data = {
            "image_directory": str(dataset_dir.absolute()),
            "batch_size": 5,
            "max_workers": 2,
            "force_rebuild": False,
            "process_faces": True,
        }

        print(f"开始构建索引: {build_data}")
        print("⚠️  这可能需要几分钟时间...")

        response = requests.post(
            f"{API_BASE_URL}/api/v1/index/build", json=build_data, timeout=300  # 5分钟超时
        )

        print(f"响应状态: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(
                f"构建完成: 处理了{result.get('processed_images', 0)}/{result.get('total_images', 0)}张图片"
            )
            print(f"耗时: {result.get('processing_time_seconds', 0):.2f}秒")
            return True
        else:
            print(f"索引构建失败: {response.text}")
            return False

    except Exception as e:
        print(f"❌ 索引构建失败: {e}")
        return False


def main():
    """运行所有测试"""
    print("🚀 开始QwenRag API功能测试")
    print("=" * 50)

    # 测试结果记录
    test_results = {}

    # 基础健康检查
    test_results["api_health"] = test_api_health()
    test_results["search_health"] = test_search_health()
    test_results["face_health"] = test_face_health()

    # 功能测试
    test_results["index_status"] = test_index_status()
    test_results["text_search"] = test_text_search()
    test_results["image_search"] = test_image_upload_and_search()
    test_results["face_detection"] = test_face_detection()

    # 可选：索引构建测试（耗时较长）
    build_test = input("\n是否测试索引构建功能？(y/N): ").lower().strip() == "y"
    if build_test:
        test_results["index_build"] = test_index_build()

    # 输出测试总结
    print("\n" + "=" * 50)
    print("📋 测试结果总结:")

    passed = 0
    total = len(test_results)

    for test_name, result in test_results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1

    print(f"\n总体结果: {passed}/{total} 个测试通过")

    if passed == total:
        print("🎉 所有测试通过！QwenRag API功能正常")
    else:
        print("⚠️  部分测试失败，请检查服务状态和配置")


if __name__ == "__main__":
    main()
