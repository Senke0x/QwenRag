# QwenRag API 部署和使用指南

## 🚀 快速开始

### 1. 启动API服务

```bash
# 方式1：直接运行
python -m api.main

# 方式2：使用uvicorn
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# 方式3：生产环境
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 2. 访问API文档

启动服务后，访问以下URL查看自动生成的API文档：

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

### 3. 健康检查

```bash
curl http://localhost:8000/health
```

## 📚 API接口说明

### 🔍 搜索API (`/api/v1/search`)

#### 文本搜索
```bash
POST /api/v1/search/text
```

示例请求：
```bash
curl -X POST "http://localhost:8000/api/v1/search/text" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "美丽的风景",
    "limit": 10,
    "similarity_threshold": 0.5
  }'
```

#### 以图搜图
```bash
POST /api/v1/search/image
```

示例请求：
```bash
curl -X POST "http://localhost:8000/api/v1/search/image" \
  -F "file=@/path/to/image.jpg" \
  -F "limit=10" \
  -F "similarity_threshold=0.5" \
  -F "include_metadata=true" \
  -F "search_faces=false"
```

### 👤 人脸识别API (`/api/v1/faces`)

#### 人脸检测
```bash
POST /api/v1/faces/detect
```

示例请求：
```bash
curl -X POST "http://localhost:8000/api/v1/faces/detect" \
  -F "file=@/path/to/face_image.jpg"
```

#### 人脸搜索
```bash
POST /api/v1/faces/search
```

示例请求：
```bash
curl -X POST "http://localhost:8000/api/v1/faces/search" \
  -F "file=@/path/to/query_face.jpg" \
  -F "limit=10" \
  -F "similarity_threshold=0.8" \
  -F "similarity_method=cosine_similarity"
```

#### 人脸比较
```bash
POST /api/v1/faces/compare
```

示例请求：
```bash
curl -X POST "http://localhost:8000/api/v1/faces/compare" \
  -H "Content-Type: application/json" \
  -d '{
    "face_id_1": "image_123_face_0_abc123",
    "face_id_2": "image_456_face_1_def456",
    "similarity_method": "cosine_similarity"
  }'
```

### 📊 索引管理API (`/api/v1/index`)

#### 构建索引
```bash
POST /api/v1/index/build
```

示例请求：
```bash
curl -X POST "http://localhost:8000/api/v1/index/build" \
  -H "Content-Type: application/json" \
  -d '{
    "image_directory": "/path/to/images",
    "batch_size": 10,
    "max_workers": 4,
    "force_rebuild": false,
    "process_faces": true
  }'
```

#### 查看索引状态
```bash
GET /api/v1/index/status
```

示例请求：
```bash
curl http://localhost:8000/api/v1/index/status
```

#### 添加图片
```bash
POST /api/v1/index/add
```

示例请求：
```bash
curl -X POST "http://localhost:8000/api/v1/index/add" \
  -H "Content-Type: application/json" \
  -d '{
    "image_paths": [
      "/path/to/new_image1.jpg",
      "/path/to/new_image2.jpg"
    ],
    "process_faces": true
  }'
```

## 🧪 测试API功能

使用提供的测试脚本：

```bash
python test_api_demo.py
```

此脚本会测试所有主要API功能：
- ✅ API健康检查
- ✅ 搜索服务健康检查
- ✅ 人脸识别服务健康检查
- ✅ 索引状态查询
- ✅ 文本搜索
- ✅ 图片上传和搜索
- ✅ 人脸检测
- 🔄 索引构建（可选）

## ⚙️ 配置说明

### 环境变量配置

可以通过环境变量配置API服务：

```bash
export QWEN_RAG_HOST=0.0.0.0
export QWEN_RAG_PORT=8000
export QWEN_RAG_DEBUG=false
export QWEN_RAG_ENABLE_DOCS=true
export QWEN_RAG_MAX_FILE_SIZE=10485760  # 10MB
export QWEN_RAG_DEFAULT_SIMILARITY_THRESHOLD=0.5
export QWEN_RAG_FACE_SIMILARITY_THRESHOLD=0.8
```

### 支持的图片格式

- JPEG/JPG
- PNG
- BMP
- GIF
- WebP

### 文件大小限制

- 默认最大文件大小：10MB
- 可通过 `QWEN_RAG_MAX_FILE_SIZE` 环境变量调整

## 🐳 Docker部署

### 创建Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 创建必要的目录
RUN mkdir -p data temp_uploads

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 构建和运行

```bash
# 构建镜像
docker build -t qwenrag-api .

# 运行容器
docker run -d \
  --name qwenrag-api \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/dataset:/app/dataset \
  -e QWEN_RAG_DEBUG=false \
  qwenrag-api
```

## 🔧 故障排除

### 常见问题

1. **服务无法启动**
   - 检查端口是否被占用
   - 确认所有依赖已正确安装
   - 查看日志输出

2. **人脸识别失败**
   - 确认DASHSCOPE_API_KEY已配置
   - 检查网络连接
   - 确认图片格式正确

3. **搜索结果为空**
   - 先构建索引 (`POST /api/v1/index/build`)
   - 确认数据目录路径正确
   - 检查索引状态 (`GET /api/v1/index/status`)

4. **API响应慢**
   - 增加workers数量
   - 检查向量索引大小
   - 考虑使用缓存

### 日志查看

API服务会输出详细的日志信息，包括：
- 请求响应时间
- 错误堆栈
- 处理状态

## 📈 性能优化

### 生产环境建议

1. **多进程部署**
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

2. **反向代理**
使用Nginx作为反向代理：
```nginx
upstream qwenrag_api {
    server localhost:8000;
    server localhost:8001;
    server localhost:8002;
    server localhost:8003;
}

server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://qwenrag_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

3. **缓存策略**
- 使用Redis缓存搜索结果
- 缓存人脸检测结果
- 设置适当的缓存过期时间

4. **监控和日志**
- 使用Prometheus监控API性能
- 配置结构化日志输出
- 设置告警规则

## 🔒 安全考虑

1. **API访问控制**
   - 添加API密钥验证
   - 限制IP访问范围
   - 实施请求频率限制

2. **文件安全**
   - 验证上传文件类型
   - 扫描恶意文件
   - 限制文件大小

3. **数据隐私**
   - 人脸数据加密存储
   - 定期清理临时文件
   - 遵循数据保护法规

## 📞 技术支持

如有问题，请检查：
1. API文档：http://localhost:8000/docs
2. 健康检查：http://localhost:8000/health
3. 服务日志输出
4. 测试脚本结果

---

📋 **部署检查清单**

- [ ] Python环境 ≥ 3.8
- [ ] 安装所有依赖包
- [ ] 配置DASHSCOPE_API_KEY
- [ ] 创建数据目录
- [ ] 测试API健康检查
- [ ] 构建图片索引
- [ ] 验证核心功能
- [ ] 生产环境优化
