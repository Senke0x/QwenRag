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
├── clients/           # 🌐 API客户端
├── processors/        # 🔄 业务处理逻辑
├── vector_store/      # 💾 向量存储
├── utils/            # 🛠️ 工具函数
├── schemas/          # 📋 数据模型
├── tests/            # 🧪 测试用例
├── examples/         # 📖 使用示例
├── eval/             # 📊 评估工具
└── dataset/          # 📸 测试数据 (gitignore)
```

### 代码规范
- 遵循PEP 8 Python代码风格
- 保持85%以上的测试覆盖率
- 所有模块必须包含`__init__.py`
- 使用类型注解和文档字符串

### 测试规范
- 单元测试: `tests/unit/`
- 集成测试: `tests/integration/`  
- 真实API测试: `tests/real_api/`
- 测试脚本: `scripts/`

### 配置管理
- 环境变量配置: `config/.env.example`
- YAML配置: `config/config.yaml.example`
- 配置优先级: 环境变量 > YAML > 默认值