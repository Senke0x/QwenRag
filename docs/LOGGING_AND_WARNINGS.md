# QwenRag 日志功能和Warning解决方案

## 📋 概述

本文档介绍了QwenRag项目中的pytest warnings解决方案和新增的API请求/响应日志功能。

## ⚠️ Pytest Warnings 解决方案

### 问题分析

项目在运行pytest时会出现以下warnings：

```
DeprecationWarning: builtin type SwigPyPacked has no __module__ attribute
DeprecationWarning: builtin type SwigPyObject has no __module__ attribute
DeprecationWarning: builtin type swigvarlink has no __module__ attribute
```

**问题根源**: 这些warnings来自`faiss-cpu`包，它使用SWIG绑定Python和C++代码，在Python导入的早期阶段产生。

### 解决方案

#### 方法1：使用环境变量（推荐）

```bash
# 运行单个测试
PYTHONWARNINGS=ignore::DeprecationWarning python3 -m pytest tests/unit/test_qwen_client.py -v

# 运行所有测试
PYTHONWARNINGS=ignore::DeprecationWarning python3 -m pytest -v
```

#### 方法2：修改pytest.ini配置

在`pytest.ini`中已添加：

```ini
addopts =
    --disable-warnings
    # ... 其他选项
```

#### 方法3：在shell中设置别名

```bash
# 添加到 ~/.bashrc 或 ~/.zshrc
alias pytest-clean="PYTHONWARNINGS=ignore::DeprecationWarning python3 -m pytest"

# 使用
pytest-clean tests/unit/test_qwen_client.py -v
```

## 📝 API请求/响应日志功能

### 功能特性

1. **自动日志记录**: 记录所有API请求和响应
2. **图片数据Mask**: 自动将base64图片数据替换为可读信息
3. **美观格式化**: JSON格式化显示，易于阅读
4. **可开关控制**: 支持启用/禁用日志功能
5. **数据大小显示**: 显示实际图片数据大小

### 日志输出示例

#### 图片聊天日志

```
🚀 === CHAT_WITH_IMAGE API调用 ===
📤 REQUEST:
{
  "model": "qwen-vl-max-latest",
  "messages": [
    {
      "role": "system",
      "content": [
        {
          "type": "text",
          "text": "你是一个专业的图像分析助手"
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,<BASE64_IMAGE_DATA_1224952_BYTES>"
          }
        },
        {
          "type": "text",
          "text": "请分析这张图片"
        }
      ]
    }
  ],
  "max_tokens": 2048,
  "temperature": 0.1,
  "timeout": 60
}

📥 RESPONSE:
这是一张美丽的游戏截图，显示了《最后生还者》中的场景...
=== API调用结束 ===
```

#### 文本聊天日志

```
🚀 === CHAT_WITH_TEXT API调用 ===
📤 REQUEST:
{
  "model": "qwen-vl-max-latest",
  "messages": [
    {
      "role": "system",
      "content": "你是一个友好的AI助手"
    },
    {
      "role": "user",
      "content": "你好，请简单介绍一下你自己"
    }
  ],
  "max_tokens": 2048,
  "temperature": 0.5,
  "timeout": 60
}

📥 RESPONSE:
你好！我是通义千问，一个由阿里云开发的AI助手...
=== API调用结束 ===
```

### 使用方法

#### 1. 启用日志（默认）

```python
from clients.qwen_client import QwenClient
from config import QwenVLConfig

config = QwenVLConfig(api_key="your_api_key")
client = QwenClient(qwen_config=config, enable_logging=True)  # 默认为True

# API调用会自动记录日志
result = client.chat_with_image(image_base64, "分析图片")
```

#### 2. 禁用日志

```python
# 创建不记录日志的客户端
client = QwenClient(qwen_config=config, enable_logging=False)

# API调用不会输出日志
result = client.chat_with_text("Hello")
```

#### 3. 在测试中使用

```python
# 在单元测试中查看API调用详情
def test_api_call():
    client = QwenClient(enable_logging=True)
    result = client.chat_with_image(image_base64, "测试")
    # 会在测试输出中显示完整的请求/响应
```

### 日志配置

#### 1. 配置日志级别

```python
import logging

# 设置日志级别为INFO以显示API日志
logging.basicConfig(level=logging.INFO)

# 或者只为QwenClient设置
logging.getLogger('clients.qwen_client').setLevel(logging.INFO)
```

#### 2. 自定义日志格式

```python
import logging

# 配置自定义格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## 🧪 测试命令

### 1. 运行无warnings的测试

```bash
# 单个测试文件
PYTHONWARNINGS=ignore::DeprecationWarning python3 -m pytest tests/unit/test_qwen_client.py -v -s

# 整个测试套件
PYTHONWARNINGS=ignore::DeprecationWarning python3 -m pytest -v

# 真实API测试（显示日志）
PYTHONWARNINGS=ignore::DeprecationWarning USE_REAL_API=true DASHSCOPE_API_KEY=your_key python3 -m pytest tests/real_api/test_logging_real.py -v -s
```

### 2. 日志功能演示

```bash
# Mock API演示
PYTHONWARNINGS=ignore::DeprecationWarning python3 test_logging_demo.py

# 真实API演示
PYTHONWARNINGS=ignore::DeprecationWarning USE_REAL_API=true DASHSCOPE_API_KEY=your_key python3 tests/real_api/test_logging_real.py
```

### 3. 批量测试脚本

```bash
# 运行综合测试
PYTHONWARNINGS=ignore::DeprecationWarning USE_REAL_API=true DASHSCOPE_API_KEY=your_key python3 run_real_api_tests.py
```

## 🔧 高级配置

### 1. 环境变量配置

```bash
# 设置永久的warnings过滤
export PYTHONWARNINGS=ignore::DeprecationWarning

# 设置API相关环境变量
export USE_REAL_API=true
export DASHSCOPE_API_KEY=your_api_key

# 设置日志级别
export PYTHONLOGLEVEL=INFO
```

### 2. 项目级配置

在项目根目录创建`.env`文件：

```env
PYTHONWARNINGS=ignore::DeprecationWarning
USE_REAL_API=false
DASHSCOPE_API_KEY=your_api_key_here
LOGLEVEL=INFO
```

### 3. IDE配置

#### VS Code

在`.vscode/settings.json`中添加：

```json
{
    "python.testing.pytestArgs": [
        "--tb=short",
        "-v"
    ],
    "python.envFile": "${workspaceFolder}/.env",
    "python.testing.pytestEnabled": true
}
```

#### PyCharm

1. Settings → Tools → Python Integrated Tools → Testing → Default test runner: pytest
2. Run/Debug Configurations → Environment variables → 添加 `PYTHONWARNINGS=ignore::DeprecationWarning`

## 📊 日志数据处理

### 1. 图片数据Mock

原始图片base64数据会被替换为：

```
data:image/jpeg;base64,<BASE64_IMAGE_DATA_1224952_BYTES>
```

这样既保留了格式信息，又显示了数据大小，避免了日志过长。

### 2. 数据大小计算

- 显示实际base64字符串长度
- 便于分析API调用的数据传输量
- 帮助优化图片大小

### 3. 响应内容处理

- 完整显示API响应内容
- 保持原始格式（JSON、文本等）
- 便于调试和分析

## 🚀 实际应用场景

### 1. 开发调试

```python
# 开发时启用日志，查看API交互详情
client = QwenClient(enable_logging=True)
result = client.chat_with_image(image, prompt)
# 查看请求参数是否正确，响应是否符合预期
```

### 2. 生产环境

```python
# 生产环境禁用详细日志，提高性能
client = QwenClient(enable_logging=False)
result = client.chat_with_image(image, prompt)
```

### 3. 测试验证

```python
# 在测试中验证API调用
def test_api_call():
    client = QwenClient(enable_logging=True)
    result = client.chat_with_text("test")
    # 通过日志验证请求格式和响应内容
```

### 4. 性能分析

通过日志可以分析：
- API调用频率
- 请求数据大小
- 响应时间（结合时间戳）
- 参数配置效果

## 📈 性能影响

### 1. 日志开销

- 启用日志会增加少量CPU和内存开销
- 主要用于开发和调试阶段
- 生产环境建议禁用详细日志

### 2. 网络传输

- 日志不影响实际的API网络传输
- 只是本地处理和显示
- 图片base64 mock不会发送给API

### 3. 存储空间

- 日志文件可能较大（包含完整请求/响应）
- 建议配置日志轮转
- 定期清理历史日志

## 🔍 故障排除

### 1. 日志不显示

```python
# 检查日志级别
import logging
logging.getLogger('clients.qwen_client').setLevel(logging.DEBUG)

# 检查enable_logging参数
client = QwenClient(enable_logging=True)
```

### 2. Warnings仍然出现

```bash
# 确保环境变量设置正确
echo $PYTHONWARNINGS

# 使用更强制的方法
python3 -W ignore::DeprecationWarning -m pytest
```

### 3. 图片数据显示异常

- 检查图片base64格式是否正确
- 确认图片文件存在且可读
- 验证base64编码是否完整

## 📚 总结

通过本次更新：

1. ✅ **解决了Warnings问题**: 使用环境变量过滤SWIG相关的DeprecationWarning
2. ✅ **增加了日志功能**: 自动记录所有API请求和响应
3. ✅ **实现了数据Mock**: 图片base64数据被替换为可读信息
4. ✅ **提供了开关控制**: 支持启用/禁用日志功能
5. ✅ **优化了输出格式**: 美观的JSON格式化显示

这些改进让开发和调试变得更加便利，同时保持了生产环境的性能。
