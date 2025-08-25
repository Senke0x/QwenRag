# Qwen Rerank API 集成方案

## 📋 API 概述

基于第一阶段重排序需求，分析Qwen Rerank API的集成方案和技术参数。

### 支持的Rerank模型
1. **DashScope gte-rerank**: 阿里云官方重排序模型
2. **Qwen3-Reranker**: Hugging Face开源模型系列
   - Qwen3-Reranker-0.6B (轻量级)
   - Qwen3-Reranker-4B (平衡型)
   - Qwen3-Reranker-8B (高性能)

## 🔧 DashScope Text Reranking API

### 基本信息
- **服务地址**: `https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank`
- **请求方式**: POST
- **模型名称**: `gte-rerank-v2`
- **上下文长度**: 32K tokens
- **多语言支持**: 100+ 语言

### 请求参数详情

#### 核心参数
```json
{
  "model": "gte-rerank-v2",           // 模型名称(必需)
  "input": {
    "query": "string",                 // 查询文本(必需)
    "documents": ["string"]            // 候选文档列表(必需)
  },
  "parameters": {
    "return_documents": true,          // 是否返回原始文档(可选,默认false)
    "top_n": 5,                       // 返回前N个结果(可选,默认全部)
    "instruction": "string"           // 自定义指令(可选,可提升1-5%性能)
  }
}
```

#### 参数限制
- **单行最大Token**: 4,000 tokens (每个Query或Document)
- **最大Document数量**: 500个
- **最大总Token数**: 30,000 tokens (所有Query和Document总计)

### 响应格式
```json
{
  "output": {
    "results": [
      {
        "index": 0,                    // 原始文档索引
        "relevance_score": 0.98,       // 相关性分数[0,1]
        "document": "string"           // 原始文档(仅当return_documents=true)
      }
    ]
  },
  "usage": {
    "total_tokens": 1500              // 消耗的总Token数
  },
  "request_id": "string"
}
```

### Python SDK 调用示例
```python
import dashscope
from dashscope import TextReRank

def rerank_with_dashscope(query: str, documents: List[str], top_n: int = 10):
    """使用DashScope进行重排序"""

    response = TextReRank.call(
        model='gte-rerank-v2',
        query=query,
        documents=documents,
        top_n=top_n,
        return_documents=True
    )

    if response.status_code == 200:
        return response.output.results
    else:
        raise Exception(f"Rerank API error: {response}")
```

## 🚀 Qwen3 Reranker 本地部署方案

### 模型选择建议
```python
MODEL_RECOMMENDATIONS = {
    "development": {
        "model": "Qwen3-Reranker-0.6B",
        "memory": "4GB",
        "performance": "快速",
        "accuracy": "中等"
    },
    "production": {
        "model": "Qwen3-Reranker-8B",
        "memory": "16GB",
        "performance": "中等",
        "accuracy": "高"
    },
    "edge": {
        "model": "Qwen3-Reranker-0.6B",
        "memory": "2GB",
        "performance": "最快",
        "accuracy": "基础"
    }
}
```

### 本地部署代码示例
```python
from transformers import AutoTokenizer, AutoModel
import torch

class LocalQwenReranker:
    """本地Qwen Reranker部署"""

    def __init__(self, model_name: str = "Qwen/Qwen3-Reranker-8B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def rerank(self, query: str, documents: List[str], top_k: int = 10):
        """本地重排序"""

        # 构建输入格式
        inputs = []
        for doc in documents:
            input_text = f"<Query>: {query}\n<Document>: {doc}"
            inputs.append(input_text)

        # 批量编码
        encoded = self.tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt"
        )

        # 模型推理
        with torch.no_grad():
            outputs = self.model(**encoded)
            scores = outputs.last_hidden_state[:, 0, :].mean(dim=1)

        # 排序结果
        sorted_indices = torch.argsort(scores, descending=True)

        results = []
        for i, idx in enumerate(sorted_indices[:top_k]):
            results.append({
                "index": idx.item(),
                "score": scores[idx].item(),
                "document": documents[idx]
            })

        return results
```

## 🔄 QwenRag 集成方案

### 重排序处理器集成
```python
# processors/reranking_processor.py
class QwenRerankingProcessor:
    """Qwen重排序处理器"""

    def __init__(self, config: RerankingConfig):
        self.config = config

        if config.use_dashscope:
            self.reranker = DashScopeReranker(api_key=config.api_key)
        else:
            self.reranker = LocalQwenReranker(model_name=config.model_name)

    async def rerank_results(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_k: int = 20
    ) -> List[RetrievalResult]:
        """重排序检索结果"""

        # 提取文档文本
        documents = [result.description for result in candidates]

        # 调用重排序API
        if self.config.use_dashscope:
            rerank_results = await self._dashscope_rerank(query, documents, top_k)
        else:
            rerank_results = self._local_rerank(query, documents, top_k)

        # 重新组织结果
        reranked_candidates = []
        for result in rerank_results:
            original_candidate = candidates[result['index']]
            original_candidate.rerank_score = result['score']
            reranked_candidates.append(original_candidate)

        return reranked_candidates
```

### 配置参数设计
```python
# config/reranking_config.py
@dataclass
class RerankingConfig:
    """重排序配置"""

    # API选择
    use_dashscope: bool = True              # 使用DashScope API
    api_key: str = ""                       # API密钥

    # DashScope参数
    model_name: str = "gte-rerank-v2"       # 模型名称
    return_documents: bool = False          # 是否返回原始文档
    custom_instruction: str = ""            # 自定义指令

    # 本地模型参数
    local_model_name: str = "Qwen/Qwen3-Reranker-8B"
    device: str = "auto"                    # 设备选择
    batch_size: int = 32                    # 批处理大小

    # 性能参数
    max_candidates: int = 200               # 最大候选数
    top_k: int = 20                        # 返回结果数
    timeout: int = 30                      # 请求超时(秒)

    # 缓存配置
    enable_cache: bool = True               # 启用缓存
    cache_ttl: int = 3600                  # 缓存TTL(秒)
```

## 📊 性能对比分析

### API方案 vs 本地部署
```python
PERFORMANCE_COMPARISON = {
    "DashScope API": {
        "优势": [
            "无需本地GPU资源",
            "模型自动更新",
            "高可用性保证",
            "按用量付费"
        ],
        "劣势": [
            "网络延迟影响",
            "API调用限制",
            "数据隐私考虑",
            "长期成本较高"
        ],
        "适用场景": "快速验证、小规模应用"
    },

    "本地部署": {
        "优势": [
            "低延迟响应",
            "数据完全私有",
            "无API限制",
            "长期成本较低"
        ],
        "劣势": [
            "需要GPU资源",
            "模型更新维护",
            "部署复杂度",
            "资源管理负担"
        ],
        "适用场景": "生产环境、大规模应用"
    }
}
```

### 性能基准测试
```python
# tests/performance/test_rerank_performance.py
class TestRerankPerformance:
    """重排序性能测试"""

    def test_dashscope_latency(self):
        """DashScope API延迟测试"""
        # 目标: <100ms (不包含网络延迟)

    def test_local_model_latency(self):
        """本地模型延迟测试"""
        # 目标: <50ms

    def test_throughput_comparison(self):
        """吞吐量对比测试"""
        # DashScope: 受API限制
        # 本地模型: 受硬件限制

    def test_accuracy_comparison(self):
        """精度对比测试"""
        # 对比不同模型的重排序效果
```

## 💰 成本分析

### DashScope API计费
- **计费方式**: 按Token消耗计费
- **预估成本**:
  - 1000次重排序请求(每次50个候选): ~50-100元/月
  - 10000次重排序请求: ~500-1000元/月

### 本地部署成本
- **硬件需求**:
  - Qwen3-Reranker-8B: 16GB GPU显存
  - Qwen3-Reranker-0.6B: 4GB GPU显存
- **电力成本**: ~200-500元/月(24小时运行)

## 🎯 集成建议

### 第一阶段建议
1. **优先使用DashScope API**: 快速验证重排序效果
2. **设置降级方案**: API失败时使用简单cosine相似度
3. **性能监控**: 监控API延迟和准确率

### 生产环境建议
1. **混合部署**: 核心业务使用本地模型，边缘场景使用API
2. **模型选择**: 根据硬件条件选择合适的模型尺寸
3. **缓存策略**: 对常见查询结果进行缓存

---

*文档版本: v1.0*
*最后更新: 2025-08-25*
*技术负责人: QwenRag团队*
