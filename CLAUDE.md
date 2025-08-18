# Claude Code 项目配置

## 项目结构要求

### 文档管理规范
- 所有Claude生成的总结性文档必须放在 `docs/` 目录下
- 禁止在根目录直接创建.md文档文件
- 所有分析报告、总结文档统一归档到docs目录

### 目录结构说明
```
QwenRag/
├── docs/              # 📚 所有文档文件 (包括Claude生成的总结)
├── scripts/           # 🔧 测试和工具脚本  
├── config/            # ⚙️ 配置文件和示例
├── clients/           # 🌐 API客户端 (QwenClient, PromptManager)
├── processors/        # 🔄 业务处理逻辑 (ImageProcessor)
├── vector_store/      # 💾 向量存储 (FaissStore)
├── utils/            # 🛠️ 工具函数 (Logger, RetryUtils, ImageUtils)
├── schemas/          # 📋 数据模型 (ImageMetadata等)
├── tests/            # 🧪 测试用例 (unit/integration/real_api)
├── examples/         # 📖 使用示例
├── design/           # 🏗️ 架构设计文档
└── dataset/          # 📸 测试数据 (gitignore)
```

### 代码规范
- 遵循PEP 8 Python代码风格
- 保持85%以上的测试覆盖率
- 所有模块必须包含`__init__.py`
- 使用类型注解和文档字符串

### 测试规范
- 单元测试: `tests/unit/` (85%+覆盖率要求)
- 集成测试: `tests/integration/` (端到端流程验证)
- 真实API测试: `tests/real_api/` (生产环境验证)
- 测试脚本: `scripts/` (自动化测试工具)
- 测试标记: `@pytest.mark.unit/integration/slow`

### 配置管理
- 环境变量配置: `config/.env.example`
- YAML配置: `config/config.yaml.example`
- 根级配置: `config.py` (程序入口配置)
- 配置优先级: 环境变量 > YAML > 根级默认值

### 开发与运行命令
```bash
# 运行单元测试
pytest tests/unit/ -m unit

# 运行集成测试  
pytest tests/integration/ -m integration

# 运行真实API测试
USE_REAL_API=true pytest tests/real_api/

# 生成覆盖率报告
pytest --cov-report=html

# 运行代码质量检查
flake8 . && black --check . && isort --check .
```