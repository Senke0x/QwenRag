# QwenRag 真实API测试指南

## 概述

本文档提供了运行QwenRag真实API测试的详细指令和配置说明。测试使用真实的dataset图片数据和Qwen API。

## 环境准备

### 1. 环境变量设置

在运行真实API测试之前，需要设置以下环境变量：

```bash
# 设置API密钥
export DASHSCOPE_API_KEY="your_api_key_here"

# 启用真实API测试
export USE_REAL_API=true
```

### 2. 验证环境变量

```bash
# 检查API密钥是否设置
echo $DASHSCOPE_API_KEY

# 检查真实API标志是否启用
echo $USE_REAL_API
```

### 3. 验证dataset目录

确保项目根目录下存在`dataset/`文件夹，并包含测试图片：

```bash
# 检查dataset目录
ls -la dataset/

# 应该看到类似输出：
# The Last of Us™ Part I_20230212234856.jpg
# The Last of Us™ Part I_20230219123321.jpg
# ...等游戏截图文件
```

## 测试命令

### 1. 快速验证测试

运行基本功能验证：

```bash
# 进入项目目录
cd /Users/chaisenpeng/Document/Github/QwenRag

# 运行快速验证脚本
python tests/real_api/test_qwen_client_real.py
```

### 2. 完整真实API测试套件

运行所有真实API测试：

```bash
# 运行真实API测试目录下的所有测试
python -m pytest tests/real_api/ -v -s

# 或者指定特定测试文件
python -m pytest tests/real_api/test_qwen_client_real.py -v -s
```

### 3. 带详细输出的测试

```bash
# 运行测试并显示API响应内容
python -m pytest tests/real_api/test_qwen_client_real.py::TestQwenClientRealAPI::test_image_analysis_landscape -v -s

# 运行结构化分析测试
python -m pytest tests/real_api/test_qwen_client_real.py::TestQwenClientRealAPI::test_structured_image_analysis -v -s

# 运行批量处理测试
python -m pytest tests/real_api/test_qwen_client_real.py::TestQwenClientRealAPI::test_multiple_images_batch -v -s
```

### 4. 跳过真实API的常规测试

如果想运行常规mock测试（不调用真实API）：

```bash
# 不设置USE_REAL_API，或设置为false
export USE_REAL_API=false

# 运行常规测试
python -m pytest tests/unit/test_qwen_client.py -v
python -m pytest tests/integration/test_client_integration.py -v
```

## 测试类型说明

### 1. 客户端基础功能测试

```bash
# 测试客户端初始化和基本功能
python -c "
from tests.real_api.test_qwen_client_real import run_manual_tests
run_manual_tests()
"
```

### 2. 图片分析测试

包含以下测试场景：
- **风景图片分析**: 测试对游戏场景的理解
- **人物图片分析**: 测试人物检测和描述
- **界面图片分析**: 测试游戏界面识别
- **结构化分析**: 测试JSON格式输出

### 3. 批量处理测试

测试多张图片的连续处理能力：

```bash
# 运行批量处理测试
python -m pytest tests/real_api/test_qwen_client_real.py::TestQwenClientRealAPI::test_multiple_images_batch -v -s
```

### 4. 错误处理测试

测试API错误情况的处理：

```bash
# 测试无效API密钥
python -m pytest tests/real_api/test_qwen_client_real.py::TestQwenClientRealAPI::test_error_handling_invalid_api_key -v -s
```

## 预期输出示例

### 成功的测试输出

```
🚀 开始QwenClient真实API测试
============================================================
📁 数据集信息: {
  "dataset_exists": true,
  "total_images": 24,
  "categorized_count": 24,
  "categories": {
    "game_with_people": 6,
    "game_landscape": 6,
    "game_interface": 5,
    "game_other": 7
  }
}
✅ 客户端初始化成功
✅ 文本聊天测试通过: 你好！我是通义千问，一个由阿里云开发的AI助手...
✅ 图片分析测试通过: 这是一张游戏截图，显示了《最后生还者》中的场景...
============================================================
📊 测试结果: 2/2 通过
🎉 所有真实API测试通过！
```

### pytest详细输出

```
tests/real_api/test_qwen_client_real.py::TestQwenClientRealAPI::test_client_initialization PASSED
tests/real_api/test_qwen_client_real.py::TestQwenClientRealAPI::test_text_chat_basic PASSED
tests/real_api/test_qwen_client_real.py::TestQwenClientRealAPI::test_image_analysis_landscape PASSED
tests/real_api/test_qwen_client_real.py::TestQwenClientRealAPI::test_structured_image_analysis PASSED

✅ 客户端信息: {
  "model": "qwen-vl-max-latest",
  "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
  "max_tokens": 2000,
  "temperature": 0.1,
  "timeout": 60
}

✅ 风景图片分析 (dataset/The Last of Us™ Part I_20230219123504.jpg): 这张图片展示了一个室内场景，可能是游戏《最后生还者》中的一个场景...
```

## 故障排除

### 1. API密钥问题

```bash
# 错误：认证失败
# 解决：检查API密钥是否正确设置
echo "当前API密钥: $DASHSCOPE_API_KEY"

# 重新设置API密钥
export DASHSCOPE_API_KEY="your_actual_api_key_here"
```

### 2. 图片文件问题

```bash
# 错误：找不到图片文件
# 解决：验证dataset目录和文件
python -c "
from tests.test_data import test_data
print(test_data.verify_dataset())
"
```

### 3. 导入错误

```bash
# 错误：模块导入失败
# 解决：确保在项目根目录运行测试
pwd  # 应该显示 /Users/chaisenpeng/Document/Github/QwenRag

# 或者设置PYTHONPATH
export PYTHONPATH=/Users/chaisenpeng/Document/Github/QwenRag:$PYTHONPATH
```

### 4. 网络连接问题

```bash
# 测试网络连接
curl -I https://dashscope.aliyuncs.com

# 如果有代理问题，可能需要设置代理环境变量
export HTTP_PROXY=your_proxy
export HTTPS_PROXY=your_proxy
```

## 性能和费用注意事项

### 1. API调用费用

- 每次图片分析会消耗API quota
- 建议在开发阶段控制测试频率
- 生产环境使用前请确认费用预算

### 2. 并发限制

```bash
# 如果遇到限流错误，可以降低并发数
# 修改测试中的并发数量或添加延迟
python -c "
import time
time.sleep(1)  # 在测试间添加延迟
"
```

### 3. 超时设置

```bash
# 如果网络较慢，可以增加超时时间
# 在config.py中调整timeout参数
```

## 高级使用

### 1. 自定义测试数据

```python
# 添加自己的测试图片到dataset目录
# 然后更新 tests/test_data.py 中的分类

from tests.test_data import test_data
test_data.categorized_images["custom_category"] = ["your_image.jpg"]
```

### 2. 自定义测试用例

```python
# 在 tests/real_api/ 目录下创建新的测试文件
# 参考 test_qwen_client_real.py 的结构
```

### 3. 集成到CI/CD

```yaml
# GitHub Actions示例
- name: Run Real API Tests
  env:
    DASHSCOPE_API_KEY: ${{ secrets.DASHSCOPE_API_KEY }}
    USE_REAL_API: true
  run: |
    python -m pytest tests/real_api/ -v
```

## 总结

通过以上指令，你可以：

1. **验证环境配置**：确保API密钥和测试数据就绪
2. **运行基础测试**：验证客户端基本功能
3. **进行图片分析**：测试不同类型图片的分析能力
4. **批量处理验证**：测试系统处理多张图片的能力
5. **错误处理验证**：确保异常情况的正确处理

建议按照上述顺序逐步进行测试，确保每个环节都正常工作后再进行下一步。