# 配置文件说明

## 文件说明

- `config.yaml.example` - YAML配置文件示例
- `.env.example` - 环境变量配置示例

## 使用方法

### 1. 环境变量配置 (推荐)

```bash
# 复制环境变量示例文件
cp config/.env.example .env

# 编辑配置文件，设置你的API密钥
nano .env
```

### 2. YAML配置文件

```bash  
# 复制YAML配置示例文件
cp config/config.yaml.example config.yaml

# 编辑配置文件
nano config.yaml
```

## 配置优先级

1. 环境变量 (最高优先级)
2. YAML配置文件
3. 默认值 (最低优先级)

## 必需配置

- `DASHSCOPE_API_KEY` - Qwen API密钥

## 可选配置

- `LOG_LEVEL` - 日志级别 (DEBUG/INFO/WARNING/ERROR)
- `INPUT_DIR` - 输入图片目录
- `OUTPUT_DIR` - 输出结果目录
- `INDEX_DIR` - 索引存储目录