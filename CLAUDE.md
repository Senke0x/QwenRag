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

## 开发工作流程规范

### 智能开发工作流 (Agent-Driven Development)

本项目采用基于agents的智能化开发工作流，确保代码质量和开发效率。

#### 工作流程步骤

1. **需求分析阶段** 📋
   - Claude接收用户需求
   - 分析技术可行性和实现方案
   - 提供详细的技术方案供用户审查

2. **用户审查确认** ✅
   - 用户审查Claude提供的技术方案
   - 确认实现方向和技术选型
   - 批准进入开发阶段

3. **代码实现阶段** 💻
   - Claude根据确认的方案进行代码开发
   - 遵循项目代码规范和目录结构
   - 完成功能实现和必要的文档更新

4. **自动代码审查** 🔍
   - 自动调用 `code-reviewer` agent
   - 检查代码质量、架构设计、安全性
   - 评估测试覆盖率和文档完整性

5. **测试验证阶段** 🧪
   - 调用 `test-validator` agent
   - 运行相关单元测试和集成测试
   - 验证代码变更不会破坏现有功能

6. **提交推送阶段** 📤
   - 调用 `git-commit-pusher` agent
   - 生成规范的commit消息
   - 提交代码并推送到远程仓库

#### 异常处理机制

**Code Review异常处理**:
- 如果 `code-reviewer` 发现问题或建议改动
- Claude会向用户说明具体问题
- 询问用户是否采纳建议并进行相应修改
- 修改完成后重新进入code review流程

**测试验证异常处理**:
- 如果 `test-validator` 发现测试失败
- Claude会提供具体的失败test case清单
- 详细说明失败原因和影响范围
- 向用户询问如何处理失败的测试

**提交推送异常处理**:
- 如果 `git-commit-pusher` 遇到推送失败
- Claude会报告具体的git错误信息
- 提供可能的解决方案
- 等待用户确认处理方式

#### 工作流程控制

**必须条件**:
- 每个阶段必须成功完成才能进入下一阶段
- Code review必须通过或得到用户确认
- 测试验证必须全部通过
- 仅在所有验证通过后才执行git操作

**质量保证**:
- 保持85%以上的测试覆盖率
- 遵循PEP 8代码规范
- 确保所有修改都经过review和测试
- 提交消息规范和内容完整

**用户交互原则**:
- 重要决策点必须获得用户确认
- 异常情况必须向用户报告并询问处理方式
- 提供清晰的进度反馈和状态更新
- 保持工作流的透明性和可控性
