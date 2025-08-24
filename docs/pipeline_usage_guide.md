# QwenRag Pipeline使用指南

## 概述

QwenRag现已实现完整的端到端图像检索流水线，包括：
- **IndexingPipeline**: 图片解析、向量化存储
- **RetrievalPipeline**: 多模态检索查询
- 支持文本搜图、以图搜图、混合搜索

## 系统架构

```
📁 pipelines/
├── indexing_pipeline.py     # 索引构建流水线
├── retrieval_pipeline.py    # 检索查询流水线
└── __init__.py              # 模块导出
```

### 数据流程

```
图片目录 → ImageProcessor → EmbeddingProcessor → FAISS向量库
    ↓                                                   ↑
元数据文件 ←→ RetrievalPipeline ←→ 用户查询(文本/图片)
```

## 快速开始

### 1. 环境配置

```bash
# 安装依赖
pip install -r requirements.txt

# 设置API密钥
export DASHSCOPE_API_KEY="your_api_key_here"

# 或者创建.env文件
echo "DASHSCOPE_API_KEY=your_api_key_here" > .env
```

### 2. 准备数据

```bash
# 将图片文件放入dataset目录
mkdir -p dataset
cp /path/to/your/images/* dataset/
```

### 3. 基础使用

```python
from pipelines.indexing_pipeline import IndexingPipeline
from pipelines.retrieval_pipeline import RetrievalPipeline

# 构建索引
indexing = IndexingPipeline(
    metadata_save_path="my_index_metadata.json",
    batch_size=10,
    auto_save=True
)

results = indexing.build_index_from_directory("dataset/")
print(f"索引构建完成: 成功 {results['success']}, 失败 {results['failed']}")

# 检索查询
retrieval = RetrievalPipeline(
    metadata_path="my_index_metadata.json",
    default_top_k=5
)

# 文本查询
text_results = retrieval.search_by_text("风景照片", top_k=5)

# 图片查询
image_results = retrieval.search_by_image("query_image.jpg", top_k=5)

# 混合查询
hybrid_results = retrieval.hybrid_search(
    query_text="游戏截图",
    query_image="example.jpg",
    text_weight=0.6,
    image_weight=0.4
)
```

## 详细功能说明

### IndexingPipeline - 索引构建

#### 主要功能
- ✅ 自动扫描图片目录（支持递归）
- ✅ 批量并行处理图片
- ✅ 增量索引构建（跳过已处理文件）
- ✅ 多层次向量化（全图、描述、人脸）
- ✅ 元数据持久化存储
- ✅ 详细的处理统计

#### 配置选项

```python
IndexingPipeline(
    image_processor=None,           # 图片处理器（可选）
    embedding_processor=None,       # 向量处理器（可选）
    metadata_save_path="index_metadata.json",  # 元数据保存路径
    batch_size=10,                 # 批处理大小
    max_workers=4,                 # 最大并发数
    auto_save=True                 # 自动保存索引
)
```

#### 使用示例

```python
# 基本使用
pipeline = IndexingPipeline()
results = pipeline.build_index_from_directory("dataset/")

# 高级配置
pipeline = IndexingPipeline(
    batch_size=5,        # 小批次处理
    max_workers=2,       # 限制并发
    auto_save=False      # 手动保存
)

# 处理特定图片
image_paths = ["img1.jpg", "img2.jpg"]
batch_results = pipeline.process_image_batch(image_paths)

# 获取统计信息
stats = pipeline.get_statistics()
print(f"处理成功: {stats['success_count']}")
```

### RetrievalPipeline - 检索查询

#### 主要功能
- 🔍 **文本搜图**: 自然语言描述检索
- 🖼️ **以图搜图**: 基于图像相似度检索
- 🔗 **混合搜索**: 文本+图像联合查询
- 📊 **结果增强**: 元数据信息补充
- 🎯 **智能过滤**: 相似度阈值过滤
- 🔄 **结果去重**: 同图片多匹配合并

#### 配置选项

```python
RetrievalPipeline(
    embedding_processor=None,        # 向量处理器
    metadata_path="index_metadata.json",  # 元数据路径
    default_top_k=10,               # 默认返回数量
    similarity_threshold=0.3,       # 相似度阈值
    enable_rerank=False            # 启用重排序
)
```

#### 查询类型

##### 1. 文本查询
```python
results = retrieval.search_by_text(
    query_text="游戏截图",
    top_k=5,
    enable_filter=True,    # 启用相似度过滤
    enable_dedup=True      # 启用去重
)

for result in results:
    print(f"图片: {result['image_path']}")
    print(f"相似度: {result['similarity_score']:.3f}")
    print(f"匹配类型: {result['match_type']}")
    print(f"描述: {result['metadata']['description']}")
```

##### 2. 图片查询
```python
# 使用文件路径
results = retrieval.search_by_image("query.jpg", top_k=5)

# 使用图片bytes数据
with open("query.jpg", "rb") as f:
    image_bytes = f.read()
results = retrieval.search_by_image(image_bytes, top_k=5)
```

##### 3. 混合查询
```python
results = retrieval.hybrid_search(
    query_text="风景照片",      # 文本查询
    query_image="sample.jpg",   # 图片查询
    top_k=10,
    text_weight=0.7,           # 文本权重
    image_weight=0.3           # 图片权重
)
```

#### 结果格式

```python
{
    "vector_id": "img_123_full",           # 向量ID
    "unique_id": "img_123",                # 图片唯一ID
    "similarity_score": 0.85,              # 相似度评分
    "distance": 0.15,                      # 向量距离
    "rank": 1,                             # 排名
    "match_type": "image_full",            # 匹配类型
    "image_path": "/path/to/image.jpg",    # 图片路径
    "metadata": {                          # 元数据信息
        "description": "游戏截图...",
        "is_snap": True,
        "has_person": False,
        "timestamp": "2023-02-26 11:09:28"
    },
    "all_match_types": ["image_full", "description"]  # 所有匹配类型
}
```

## 性能优化

### 索引构建优化
```python
# 大规模数据处理
IndexingPipeline(
    batch_size=20,           # 增大批处理
    max_workers=8,           # 提高并发
    auto_save=True           # 定期保存
)

# 增量更新
pipeline.build_index_from_directory(
    "new_images/",
    resume_from_metadata=True  # 跳过已处理文件
)
```

### 检索优化
```python
# 调整相似度阈值
RetrievalPipeline(
    similarity_threshold=0.5,  # 提高阈值过滤更多结果
    default_top_k=20          # 增加候选数量
)

# 重新加载元数据
retrieval.reload_metadata()  # 更新索引后调用
```

## 监控和调试

### 统计信息
```python
# 索引构建统计
indexing_stats = indexing.get_statistics()
print(f"总处理: {indexing_stats['total_processed']}")
print(f"成功率: {indexing_stats['success_count']/indexing_stats['total_processed']:.2%}")

# 检索统计
retrieval_stats = retrieval.get_statistics()
print(f"向量数量: {retrieval_stats['vector_count']}")
print(f"元数据数: {retrieval_stats['metadata_count']}")
```

### 日志配置
```python
import logging

# 启用详细日志
logging.basicConfig(level=logging.DEBUG)

# 或者只显示重要信息
logging.basicConfig(level=logging.INFO)
```

## 故障排除

### 常见问题

1. **API密钥错误**
   ```
   ValueError: DASHSCOPE_API_KEY环境变量未设置或为空
   ```
   解决：设置正确的API密钥

2. **内存不足**
   - 降低batch_size和max_workers
   - 启用auto_save定期保存

3. **处理速度慢**
   - 检查网络连接
   - 调整并发参数
   - 检查API调用限制

4. **搜索结果为空**
   - 检查相似度阈值设置
   - 确认索引构建成功
   - 检查元数据文件完整性

### 调试技巧

```python
# 检查索引状态
stats = pipeline.get_statistics()
if stats['success_count'] == 0:
    print("没有成功处理的文件，检查API配置")

# 检查元数据
retrieval.reload_metadata()
if len(retrieval.metadata_index) == 0:
    print("元数据为空，需要重新构建索引")

# 验证单个文件处理
metadata = image_processor.process_image("test.jpg")
print(f"处理状态: {metadata.processing_status}")
```

## 集成示例

### Web API集成
```python
from flask import Flask, request, jsonify

app = Flask(__name__)
retrieval = RetrievalPipeline()

@app.route('/search/text')
def text_search():
    query = request.args.get('q')
    results = retrieval.search_by_text(query, top_k=10)
    return jsonify(results)

@app.route('/search/image', methods=['POST'])
def image_search():
    image_data = request.files['image'].read()
    results = retrieval.search_by_image(image_data, top_k=10)
    return jsonify(results)
```

### 批处理脚本
```python
#!/usr/bin/env python3
import sys
from pipelines.indexing_pipeline import IndexingPipeline

def batch_process(directory):
    pipeline = IndexingPipeline(batch_size=50, max_workers=10)
    results = pipeline.build_index_from_directory(directory)

    print(f"批处理完成:")
    print(f"- 总计: {results['total']}")
    print(f"- 成功: {results['success']}")
    print(f"- 失败: {results['failed']}")

    return results['success'] > 0

if __name__ == "__main__":
    success = batch_process(sys.argv[1])
    sys.exit(0 if success else 1)
```

## 下一步开发

### 计划功能
- [ ] 支持更多图片格式
- [ ] 分布式处理支持
- [ ] 实时索引更新
- [ ] 高级查询语法
- [ ] 结果缓存机制

### 性能优化
- [ ] 向量压缩算法
- [ ] 索引分片机制
- [ ] 异步处理支持
- [ ] GPU加速支持

---

更多技术细节请参考：
- [设计文档](../design/design.md)
- [API文档](api.md)
- [测试指南](../tests/README.md)
