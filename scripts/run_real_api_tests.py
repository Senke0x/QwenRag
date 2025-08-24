#!/usr/bin/env python3
"""
快速运行真实API测试的脚本
"""
import os
import subprocess
import sys
from pathlib import Path


def check_environment():
    """检查环境变量和依赖"""
    print("🔍 检查环境配置...")

    # 检查API密钥
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("❌ 未设置 DASHSCOPE_API_KEY 环境变量")
        print("   请执行: export DASHSCOPE_API_KEY='your_api_key'")
        return False
    else:
        print(f"✅ API密钥已设置: {api_key[:10]}...")

    # 检查USE_REAL_API标志
    use_real_api = os.getenv("USE_REAL_API", "false").lower()
    if use_real_api != "true":
        print("❌ 未启用真实API测试")
        print("   请执行: export USE_REAL_API=true")
        return False
    else:
        print("✅ 真实API测试已启用")

    # 检查dataset目录
    project_root = Path(__file__).parent
    dataset_dir = project_root / "dataset"

    if not dataset_dir.exists():
        print("❌ dataset目录不存在")
        return False

    image_files = list(dataset_dir.glob("*.jpg"))
    if len(image_files) == 0:
        print("❌ dataset目录中没有图片文件")
        return False

    print(f"✅ dataset目录包含 {len(image_files)} 张图片")

    return True


def run_quick_test():
    """运行快速验证测试"""
    print("\n🚀 运行快速验证测试...")

    try:
        # 运行基本功能测试，添加环境变量过滤warnings
        env = os.environ.copy()
        env["PYTHONWARNINGS"] = "ignore::DeprecationWarning"

        result = subprocess.run(
            [sys.executable, "tests/real_api/test_qwen_client_real.py"],
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
        )

        if result.returncode == 0:
            print("✅ 快速测试通过")
            print(result.stdout)
        else:
            print("❌ 快速测试失败")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False

    except subprocess.TimeoutExpired:
        print("❌ 测试超时（5分钟）")
        return False
    except Exception as e:
        print(f"❌ 运行测试时出错: {e}")
        return False

    return True


def run_pytest_tests():
    """运行pytest真实API测试"""
    print("\n🧪 运行完整的pytest测试套件...")

    # 设置环境变量
    env = os.environ.copy()
    env["PYTHONWARNINGS"] = "ignore::DeprecationWarning"

    test_commands = [
        # 基础功能测试
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/real_api/test_qwen_client_real.py::TestQwenClientRealAPI::test_client_initialization",
            "-v",
            "-s",
        ],
        # 文本聊天测试
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/real_api/test_qwen_client_real.py::TestQwenClientRealAPI::test_text_chat_basic",
            "-v",
            "-s",
        ],
        # 图片分析测试
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/real_api/test_qwen_client_real.py::TestQwenClientRealAPI::test_image_analysis_landscape",
            "-v",
            "-s",
        ],
        # 结构化分析测试
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/real_api/test_qwen_client_real.py::TestQwenClientRealAPI::test_structured_image_analysis",
            "-v",
            "-s",
        ],
    ]

    passed = 0
    total = len(test_commands)

    for i, cmd in enumerate(test_commands, 1):
        test_name = cmd[3].split("::")[-1]
        print(f"\n📋 运行测试 {i}/{total}: {test_name}")

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=180, env=env
            )

            if result.returncode == 0:
                print(f"✅ {test_name} 通过")
                passed += 1
                # 显示部分输出
                lines = result.stdout.split("\n")
                for line in lines[-10:]:
                    if line.strip() and not line.startswith("="):
                        print(f"   {line}")
            else:
                print(f"❌ {test_name} 失败")
                print(f"   错误: {result.stderr}")

        except subprocess.TimeoutExpired:
            print(f"❌ {test_name} 超时")
        except Exception as e:
            print(f"❌ {test_name} 出错: {e}")

    print(f"\n📊 pytest测试结果: {passed}/{total} 通过")
    return passed == total


def run_batch_test():
    """运行批量处理测试"""
    print("\n📦 运行批量处理测试...")

    try:
        # 设置环境变量
        env = os.environ.copy()
        env["PYTHONWARNINGS"] = "ignore::DeprecationWarning"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/real_api/test_qwen_client_real.py::TestQwenClientRealAPI::test_multiple_images_batch",
                "-v",
                "-s",
            ],
            capture_output=True,
            text=True,
            timeout=600,
            env=env,
        )  # 10分钟超时

        if result.returncode == 0:
            print("✅ 批量处理测试通过")
            # 显示输出
            lines = result.stdout.split("\n")
            for line in lines:
                if "批量处理" in line or "✅" in line:
                    print(f"   {line}")
        else:
            print("❌ 批量处理测试失败")
            print(f"   错误: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("❌ 批量处理测试超时（10分钟）")
        return False
    except Exception as e:
        print(f"❌ 批量处理测试出错: {e}")
        return False

    return True


def main():
    """主函数"""
    print("🎯 QwenRag 真实API测试运行器")
    print("=" * 50)

    # 检查环境
    if not check_environment():
        print("\n❌ 环境检查失败，请修复上述问题后重试")
        return False

    # 运行测试
    success_count = 0
    total_tests = 3

    # 1. 快速验证测试
    if run_quick_test():
        success_count += 1

    # 2. pytest测试套件
    if run_pytest_tests():
        success_count += 1

    # 3. 批量处理测试
    if run_batch_test():
        success_count += 1

    # 总结
    print("\n" + "=" * 50)
    print(f"📊 总体测试结果: {success_count}/{total_tests} 通过")

    if success_count == total_tests:
        print("🎉 所有真实API测试成功！")
        print("\n💡 接下来可以:")
        print("   - 运行更多详细测试: python -m pytest tests/real_api/ -v -s")
        print("   - 查看测试文档: cat REAL_API_TESTING.md")
        print("   - 集成到你的工作流程中")
        return True
    else:
        print(f"❌ 有 {total_tests - success_count} 个测试失败")
        print("\n🔧 故障排除:")
        print("   - 检查网络连接")
        print("   - 确认API密钥有效")
        print("   - 查看详细错误信息")
        print("   - 参考 REAL_API_TESTING.md 文档")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
