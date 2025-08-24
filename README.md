# QwenRag - Qwen VL图像检索系统

基于Qwen VL的智能图像检索和向量存储系统，支持文字描述搜图和以图搜图功能。

## 🚀 功能特性

- **图像智能分析**: 使用Qwen VL识别图片内容、分类、人脸检测
- **向量存储**: 基于FAISS的高效向量存储和搜索
- **多种搜索方式**:
  - 文字描述搜图
  - 以图搜图
  - 人脸相似度搜索
- **完善的错误处理**: 重试机制、错误恢复、状态跟踪
- **测试驱动开发**: 高覆盖率测试，支持真实API和模拟测试
- **命令行工具**: 易于使用的索引和搜索工具

## 📋 系统要求

- Python 3.8+
- Qwen API密钥 (DASHSCOPE_API_KEY)
- 支持的图片格式: JPG, PNG, WebP

## 🛠️ 安装

1. 克隆项目
```bash
git clone https://github.com/your-repo/QwenRag.git
cd QwenRag
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 配置API密钥
```bash
cp .env.example .env
# 编辑.env文件，设置DASHSCOPE_API_KEY
```

## 🎯 快速开始

### 1. 图片索引

```bash
# 基本用法
python main_index.py --input_dir ./images --output_dir ./index --api_key your_api_key

# 使用环境变量
export DASHSCOPE_API_KEY=your_api_key
python main_index.py --input_dir ./images --output_dir ./index

# 批量处理
python main_index.py -i ./images -o ./index -b 20 --log_level DEBUG
```

### 2. 图片搜索

```bash
# 文字搜索
python main_search.py --index_dir ./index --query "美丽的山水风景" --top_k 5

# 以图搜图
python main_search.py --index_dir ./index --image_query ./query.jpg --top_k 3

# 显示相似度分数
python main_search.py -i ./index -q "风景" --show_similarity --output_format json
```

## 🧪 运行测试

### 模拟测试 (推荐)
```bash
pytest tests/ -v
```

### 真实API测试
```bash
export USE_REAL_API=true
export DASHSCOPE_API_KEY=your_api_key
pytest tests/ -v
```

### 测试覆盖率
```bash
pytest tests/ --cov-report=html
```

## 📁 项目结构

```
QwenRag/
├── main_index.py         # 图片索引工具
├── main_search.py        # 图片搜索工具
├── config.py             # 配置管理
├── requirements.txt      # 依赖包
├──
├── schemas/              # 数据模型
│   ├── __init__.py
│   └── data_models.py
├──
├── processors/           # 图片处理器
│   ├── __init__.py
│   └── image_processor.py
├──
├── vector_store/         # 向量存储
│   ├── __init__.py
│   └── faiss_store.py
├──
├── utils/                # 工具函数
│   ├── __init__.py
│   ├── logger.py
│   ├── retry_utils.py
│   └── image_utils.py
├──
└── tests/                # 测试用例
    ├── conftest.py
    └── unit/
        ├── test_image_processor.py
        └── test_faiss_store.py
```

## ⚙️ 配置

支持YAML配置文件，参考 `config.yaml.example`:

```yaml
qwen_vl:
  api_key: "your_api_key"
  model: "qwen-vl-max-latest"
  timeout: 60

image_processor:
  supported_formats: [".jpg", ".png", ".webp"]
  max_image_size: 20971520  # 20MB
  max_resolution: [4096, 4096]

logging:
  level: "INFO"
  file: "qwen_rag.log"
```

## 🔧 核心功能

### 图片处理流程

1. **格式验证**: 检查图片格式和大小
2. **内容分析**: 使用Qwen VL识别图片内容
3. **特征提取**: 生成图片描述和标签
4. **人脸检测**: 检测和定位人脸区域
5. **向量化**: 将特征转换为向量存储

### 搜索流程

1. **查询处理**: 解析文字或图片查询
2. **向量搜索**: 在FAISS索引中搜索相似向量
3. **结果融合**: 合并语义和人脸搜索结果
4. **重排序**: 基于多种特征重新排序
5. **结果返回**: 返回最相关的图片

## 📊 数据结构

### ImageMetadata
```python
@dataclass
class ImageMetadata:
    path: str                    # 图片路径
    is_snap: bool               # 是否为截图
    is_landscape: bool          # 是否为风景照
    description: str            # 图片描述
    has_person: bool           # 是否包含人物
    face_rects: List[Tuple]    # 人脸位置框
    timestamp: str             # 时间戳
    unique_id: str             # 唯一ID
    processing_status: Enum    # 处理状态
```

## 🔍 搜索示例

### 文字搜索
```bash
python main_search.py -i ./index -q "山水风景" -k 5 --show_similarity
```

### 图片搜索
```bash
python main_search.py -i ./index --image_query sunset.jpg -k 3 -f json
```

### 搜索结果格式
```json
[
  {
    "path": "/path/to/image1.jpg",
    "description": "美丽的山水风景，夕阳西下",
    "is_landscape": true,
    "has_person": false,
    "similarity_score": 0.892
  }
]
```

## 🚨 错误处理

- **API限流**: 自动重试，指数退避
- **网络异常**: 连接超时重试
- **图片损坏**: 跳过并记录错误
- **认证失败**: 立即停止，不重试

## 📈 性能优化

- **批量处理**: 支持批量图片处理
- **内存管理**: 大图片自动压缩
- **向量索引**: FAISS高效搜索
- **并发处理**: 多线程处理支持

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📝 开发注意事项

- 遵循TDD开发模式
- 保持高测试覆盖率
- 添加完善的错误处理
- 文档和代码同步更新

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 🙏 致谢

- [Qwen VL](https://qwen.readthedocs.io/) - 强大的视觉语言模型
- [FAISS](https://github.com/facebookresearch/faiss) - 高效的向量搜索库
- [OpenAI Python SDK](https://github.com/openai/openai-python) - API客户端库

## 📞 支持

如有问题请提交 [Issue](https://github.com/your-repo/QwenRag/issues) 或联系维护者。
