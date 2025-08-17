# Qwen RAG 图像检索系统设计文档

## 需求
1. 给出 1w 张图片，能够通过文字描述进行搜图，或者进行以图搜图，这里以图搜图是基于人脸进行比对。
2. 图片集中，多以山水，人物照片为主，不包括手机截图，文件照片等信息。

## 开发 Rule
1. 语言上使用 Python3.8+
2. 对于模型的调用都是基于 Qwen相关的 API 接口来实现
   1. https://www.alibabacloud.com/help/en/model-studio/use-qwen-by-calling-api?spm=a2c63.p38356.help-menu-2400256.d_2_1_0.15ce73b516LPVm#8e7db3cf213aa
3. 开发注意高内聚 低耦合方便扩展
4. 代码格式参考下面给出的仓库，同时尽量使用下面两个仓库中用到的一些依赖 例如 tqdm asyncio 等
5. **测试驱动开发（TDD）**：先写测试用例，再实现功能
6. **错误处理优先**：所有关键操作都要有完善的错误处理和重试机制

## 参考
1. https://github.com/WarmneoN/QwenVL-Batch-OCR/blob/main/qwen-vl-ocr.py
2. https://github.com/Senke0x/RAG-Challenge-2

## 架构设计
``` mermaid
graph LR

  subgraph indexing [数据入库 - Indexing Process]
      A[图片数据] --> B[Qwen VL识别];
      B --> C[获取描述信息];
      B --> D{是否含有<br/>人脸};
      D -- 是 --> E[抽离人脸小图];

      subgraph embedding_stage_1 [Embedding Stage 1]
          direction LR
          C --> F1[Qwen<br/>Embedding];
          E --> F1;
      end

      F1 --> G[语义 embedding];
      F1 --> H[人脸 embedding];

      G --> I[(FAISS)];
      H --> I;
  end

  subgraph retrieval [数据检索 - Retrieval Process]
      J[Query] --> K[Qwen VL识别];
      K --> L[获取描述信息];
      K --> M{是否含有<br/>人脸};
      M -- 是 --> N[抽离人脸小图];

      subgraph embedding_stage_2 [Embedding Stage 2]
          direction LR
          L --> F2[Qwen<br/>Embedding];
          N --> F2;
      end

      F2 --> P[语义 Search];
      F2 --> Q[人脸 Search];

      P --> R[(FAISS)];
      Q --> R;
      R --> S[结果 merge];
      S --> T[rerank<br/>基于原图的 embedding];
      T --> U[检索完成];
  end

  %% --- Styling to match the original diagram ---
  style B fill:#d4edda,stroke:#155724
  style K fill:#d4edda,stroke:#155724
  style F1 fill:#d4edda,stroke:#155724
  style F2 fill:#d4edda,stroke:#155724
  style D fill:#fff3cd,stroke:#856404
  style M fill:#fff3cd,stroke:#856404
```

## 详细设计

### 1. 图片识别
- 输入一个绝对路径，对当前路径扫描所有的 jpg 和 png 获取图片
- 将所有数据通过 QwenVL 判断是否是手机截图，是否是风景照，是否有人，是否是多人， 构造核心数据结构
```json
{
    "path": "xx", // 原图绝对路径
    "is_snap": true, // 是否是手机截图
    "is_landscape": true, // 是否是风景照
    "description": "", // 对当前的描述，用于语义检索
    "has_person": true, // 是否有人
    "face_rects": [[x,y,w,h], [x,y,w,h]], // 对应人的框，人脸 + 人体，支持多人
    "timestamp": "", // 照片的时间戳
    "unique_id": "xx", // 获取唯一 ID
    "processing_status": "success|failed|retrying", // 处理状态
    "error_message": "", // 错误信息
    "retry_count": 0, // 重试次数
    "last_processed": "2024-01-01T00:00:00Z" // 最后处理时间
}
```

### 2. 数据 embedding
- 对于输入的结构体， 直接进行 embedding，并且如果存在 face_rects，将对应的人脸小图扣出来，将其 base64 结果直接进行 embedding，对于两个结果，直接插入到对应的 FAISS 库中，暂时使用 FLAT 进行建库（数据多了考虑上 ANN）

### 3. 检索
- 参考步骤 1 对图片进行识别
- 参考步骤 2，进行 embedding，然后分别对应的 FAISS 库进行检索，取 top10，对结果进行 merge，这里数量小于等于 20
- 将上一个步骤的结构体，拿到对应的原图，拼凑 Query 的 string，给到 rerank 模型
- 返回最终排序结果

## 错误处理与重试机制

### 1. API 调用重试策略
```python
# 重试配置
RETRY_CONFIG = {
    "max_retries": 3,
    "base_delay": 1.0,  # 基础延迟（秒）
    "max_delay": 60.0,  # 最大延迟（秒）
    "exponential_base": 2,  # 指数退避基数
    "retryable_errors": [
        "rate_limit_exceeded",
        "service_unavailable",
        "timeout",
        "connection_error"
    ]
}
```

### 2. 错误分类与处理策略
- **可重试错误**：网络超时、API限流、服务暂时不可用
- **不可重试错误**：认证失败、参数错误、图片格式不支持
- **部分成功处理**：某些图片处理失败，记录错误但继续处理其他图片

### 3. 状态管理
- 每个图片都有处理状态跟踪
- 失败任务进入重试队列
- 支持手动重试和自动重试

### 4. 日志记录
- 详细的操作日志
- 错误堆栈信息
- 性能指标记录
- 支持日志级别配置

## 测试用例设计

### 1. 单元测试

#### 1.1 图片识别模块测试
```python
# test_image_processor.py
class TestImageProcessor:
    def test_image_format_validation(self):
        """测试图片格式验证"""
        # 测试支持的格式：jpg, png
        # 测试不支持的格式：gif, bmp

    def test_qwen_vl_api_call(self):
        """测试Qwen VL API调用"""
        # 测试正常调用
        # 测试API限流
        # 测试网络超时
        # 测试认证失败

    def test_face_detection(self):
        """测试人脸检测"""
        # 测试单人脸图片
        # 测试多人脸图片
        # 测试无人脸图片
        # 测试人脸坐标准确性

    def test_image_classification(self):
        """测试图片分类"""
        # 测试风景照识别
        # 测试人物照识别
        # 测试手机截图识别
        # 测试混合内容图片

    def test_error_handling(self):
        """测试错误处理"""
        # 测试损坏图片处理
        # 测试超大图片处理
        # 测试API调用失败
        # 测试网络异常
```

#### 1.2 FAISS 向量存储测试
```python
# test_faiss_store.py
class TestFaissStore:
    def test_index_creation(self):
        """测试索引创建"""
        # 测试空索引创建
        # 测试带数据索引创建
        # 测试索引参数配置

    def test_vector_insertion(self):
        """测试向量插入"""
        # 测试单条插入
        # 测试批量插入
        # 测试重复ID处理
        # 测试异常数据插入

    def test_vector_search(self):
        """测试向量搜索"""
        # 测试精确搜索
        # 测试相似度搜索
        # 测试Top-K检索
        # 测试空结果处理

    def test_index_persistence(self):
        """测试索引持久化"""
        # 测试索引保存
        # 测试索引加载
        # 测试索引更新
        # 测试索引损坏恢复
```

#### 1.3 Embedding 模型测试
```python
# test_embedding_model.py
class TestEmbeddingModel:
    def test_text_embedding(self):
        """测试文本embedding"""
        # 测试中文文本
        # 测试英文文本
        # 测试混合语言
        # 测试空文本

    def test_image_embedding(self):
        """测试图片embedding"""
        # 测试正常图片
        # 测试人脸小图
        # 测试不同尺寸图片
        # 测试图片格式转换

    def test_api_error_handling(self):
        """测试API错误处理"""
        # 测试认证失败
        # 测试参数错误
        # 测试服务不可用
        # 测试重试机制
```

### 2. 集成测试

#### 2.1 端到端流程测试
```python
# test_integration.py
class TestIntegration:
    def test_full_indexing_pipeline(self):
        """测试完整索引流程"""
        # 测试从图片扫描到FAISS入库的完整流程
        # 测试错误恢复机制
        # 测试部分失败处理

    def test_full_retrieval_pipeline(self):
        """测试完整检索流程"""
        # 测试文字描述搜索
        # 测试以图搜图
        # 测试人脸搜索
        # 测试结果排序和rerank

    def test_error_scenarios(self):
        """测试错误场景"""
        # 测试网络中断恢复
        # 测试API服务降级
        # 测试磁盘空间不足
        # 测试内存不足
```

### 3. 性能测试（可选）

#### 3.1 批量处理测试
```python
# test_performance.py
class TestPerformance:
    def test_batch_processing(self):
        """测试批量处理性能"""
        # 测试100张图片处理时间
        # 测试1000张图片处理时间
        # 测试内存使用情况
        # 测试CPU使用情况

    def test_search_performance(self):
        """测试搜索性能"""
        # 测试不同索引大小的搜索时间
        # 测试并发搜索性能
        # 测试内存占用
```

### 4. 测试数据准备

#### 4.1 测试图片集
- **正常图片**：10张风景照、10张人物照、10张混合内容图片
- **边界情况**：超大图片、极小图片、损坏图片、不支持格式
- **特殊内容**：多人脸图片、复杂场景图片、低质量图片

#### 4.2 Mock API 服务
- 模拟Qwen VL API响应
- 模拟各种错误情况
- 模拟网络延迟和超时

#### 4.3 测试环境配置
```python
# test_config.py
TEST_CONFIG = {
    "test_image_dir": "tests/test_images/",
    "mock_api_enabled": True,
    "test_faiss_path": "tests/test_index/",
    "log_level": "DEBUG",
    "cleanup_after_test": True
}
```

## 代码架构
```
rag_image_system/
├── main_index.py             # 运行数据索引流程的入口脚本
├── main_search.py            # 运行检索流程的入口脚本
├── config.py                 # 配置文件，存放所有路径和参数
├── requirements.txt          # 项目依赖库列表
├── pytest.ini               # pytest配置文件
│
├── schemas/
│   └── data_models.py        # 定义核心数据结构 (ImageMetadata)
│
├── processors/
│   └── image_processor.py    # 封装 Qwen-VL 的图像识别和分析逻辑
│
├── models/
│   ├── embedding_model.py    # 封装 Qwen Embedding 模型的调用
│   └── rerank_model.py       # 封装 Rerank 模型的调用
│
├── vector_store/
│   └── faiss_store.py        # 封装 FAISS 数据库的增、查、存、读操作
│
├── pipelines/
│   ├── indexing_pipeline.py  # 编排整个数据索引的流程
│   └── retrieval_pipeline.py # 编排整个检索查询的流程
│
├── utils/
│   ├── file_utils.py         # 文件系统相关的辅助函数 (如扫描图片)
│   ├── image_utils.py        # 图像处理相关的辅助函数 (如裁剪人脸)
│   ├── retry_utils.py        # 重试机制工具函数
│   └── logger.py             # 日志配置和管理
│
├── tests/                    # 测试目录
│   ├── unit/                 # 单元测试
│   │   ├── test_image_processor.py
│   │   ├── test_faiss_store.py
│   │   └── test_embedding_model.py
│   ├── integration/          # 集成测试
│   │   └── test_integration.py
│   ├── performance/          # 性能测试（可选）
│   │   └── test_performance.py
│   ├── test_images/          # 测试图片
│   ├── test_data/            # 测试数据
│   └── conftest.py           # pytest配置文件
│
└── docs/                     # 文档目录
    ├── api.md                # API接口文档
    ├── deployment.md         # 部署指南
    └── troubleshooting.md    # 故障排除指南
```

## 开发流程

### 1. 测试驱动开发流程
1. **编写测试用例**：先定义期望的行为和边界情况
2. **运行测试**：确认测试失败（Red）
3. **实现功能**：编写最小代码使测试通过（Green）
4. **重构优化**：改进代码质量，保持测试通过（Refactor）

### 2. 错误处理开发流程
1. **识别错误场景**：分析可能的失败点
2. **编写错误测试**：测试异常情况的处理
3. **实现错误处理**：添加try-catch和重试逻辑
4. **验证错误恢复**：确保系统能从错误中恢复

### 3. 集成测试流程
1. **组件测试**：确保各个模块独立工作
2. **接口测试**：验证模块间的数据传递
3. **端到端测试**：验证完整业务流程
4. **错误场景测试**：验证系统在异常情况下的表现

## 质量保证

### 1. 代码覆盖率
- 目标：单元测试覆盖率 > 90%
- 工具：pytest-cov
- 报告：HTML格式的覆盖率报告

### 2. 代码质量
- 工具：flake8, black, isort
- 配置：统一的代码风格规范
- 检查：CI/CD流水线中的自动检查

### 3. 文档完整性
- API文档：使用docstring生成
- 架构文档：保持与代码同步
- 部署文档：详细的部署步骤
- 故障排除：常见问题和解决方案

## 下一步行动计划

### 1. 技术调研阶段
- [ ] 验证Qwen VL API的人脸检测能力
- [ ] 调研人脸识别和embedding的最佳实践
- [ ] 确定FAISS索引类型和参数配置

### 2. 测试用例编写阶段
- [ ] 编写图片识别模块的测试用例
- [ ] 编写FAISS存储模块的测试用例
- [ ] 编写Embedding模型的测试用例
- [ ] 编写集成测试用例

### 3. 核心功能实现阶段
- [ ] 实现图片识别和分类功能
- [ ] 实现人脸检测和裁剪功能
- [ ] 实现向量存储和检索功能
- [ ] 实现错误处理和重试机制

### 4. 系统集成和优化阶段
- [ ] 集成各个模块
- [ ] 端到端测试和调试
- [ ] 性能优化和错误处理完善
- [ ] 文档编写和部署准备
