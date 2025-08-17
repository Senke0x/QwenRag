# 测试说明

## 环境配置

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 配置环境变量：
```bash
cp .env.example .env
# 编辑.env文件，设置你的DASHSCOPE_API_KEY
```

## 运行测试

### 使用模拟API测试（默认）
```bash
pytest tests/ -v
```

### 使用真实API测试
```bash
export USE_REAL_API=true
export DASHSCOPE_API_KEY=your_api_key_here
pytest tests/ -v
```

### 运行特定测试
```bash
# 只运行图片处理器测试
pytest tests/unit/test_image_processor.py -v

# 只运行FAISS存储测试
pytest tests/unit/test_faiss_store.py -v

# 运行真实API测试
pytest tests/unit/test_image_processor.py::TestImageProcessor::test_qwen_vl_api_call_success_real -v
```

### 查看测试覆盖率
```bash
pytest tests/ --cov-report=html
# 在htmlcov/index.html中查看覆盖率报告
```

## 测试说明

### 图片处理器测试 (`test_image_processor.py`)
- **模拟测试**：使用mock API响应，测试各种场景
- **真实API测试**：调用实际的Qwen VL API（需要API密钥）
- 测试功能：格式验证、API调用、人脸检测、图片分类、错误处理等

### FAISS存储测试 (`test_faiss_store.py`)
- 测试向量存储的所有核心功能
- 索引创建、向量插入、搜索、持久化等
- 不需要外部API，完全离线测试

## 项目结构

```
QwenRag/
├── schemas/           # 数据模型
├── processors/        # 图片处理器
├── vector_store/      # FAISS向量存储
├── utils/            # 工具函数
├── config.py         # 配置管理
├── tests/            # 测试用例
└── requirements.txt  # 依赖包
```

## 注意事项

1. 真实API测试需要有效的DASHSCOPE_API_KEY
2. 真实API测试可能产生费用，建议先使用模拟测试
3. 部分测试需要网络连接
4. 如果测试失败，检查API密钥和网络连接