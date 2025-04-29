# 源代码嵌入向量生成与检索工具

这是一个用于处理源代码文件、生成嵌入向量并实现语义搜索的工具。本工具使用 [fastapi-admin](https://github.com/fastapi-admin/fastapi-admin) 作为演示源代码，将其转换为向量形式并存储在 Milvus 向量数据库中，以便进行高效的语义搜索。

## 功能特点

- **自动处理代码文件**：自动遍历指定目录下的所有代码文件
- **增量更新**：使用 Redis 缓存文件 MD5，只处理变更的文件
- **智能分块处理**：对于大型文件，使用 AST 解析将代码分解为更小的语义单元
- **向量化存储**：将代码片段转换为嵌入向量并存储在 Milvus 向量数据库中
- **语义搜索**：支持通过自然语言查询找到相关代码片段
- **路径查询**：支持通过文件路径查询相关代码

## 环境要求

- Python 3.9+
- Redis 服务
- Milvus 向量数据库
- BAAI/bge-m3 嵌入模型



## 配置说明

在使用前，需要确保：

1. Redis 服务可用（默认连接到 172.16.11.105:16379）
2. 有足够的磁盘空间用于存储 Milvus 数据库文件
3. 有权限访问和下载 BAAI/bge-m3 模型

## 使用方法

### 基本用法

```python
from main import SourceCodeEmbedding

# 初始化源代码嵌入处理类
directory_path = "demo_src"  # fastapi-admin 源码目录
file_extension = "py"        # 要处理的文件扩展名
embedding_model = "BAAI/bge-m3"  # 使用的嵌入模型

# 创建嵌入处理器实例
source_code_embedding = SourceCodeEmbedding(
    embedding_model, directory_path, file_extension
)

# 处理源代码并生成向量
source_code_embedding.run()

# 通过语义搜索查询相关代码
results = source_code_embedding.search_milvus("REDIS的默认配置")

# 通过路径查询相关代码
results = source_code_embedding.query_milvus("fastapi_admin/routes")

# 删除特定路径的数据
source_code_embedding.delete_milvus("fastapi_admin/routes/login.py")
```

## 代码结构

- `SourceCodeEmbedding`: 主类，处理源代码并生成嵌入向量
  - `__init__`: 初始化方法，设置模型、路径等参数
  - `_init_redis`: 初始化 Redis 连接
  - `_init_milvus`: 初始化 Milvus 向量数据库
  - `run`: 主处理流程，遍历所有文件并处理
  - `search_milvus`: 语义搜索方法
  - `query_milvus`: 通过路径查询方法
  - `delete_milvus`: 删除数据方法

## 性能优化

- 使用 Redis 缓存文件 MD5，避免重复处理未变更文件
- 对大文件使用 AST 解析分块处理，避免超出模型 token 限制
- 使用 Milvus 向量数据库的高效索引机制，提升检索性能

## 注意事项

- 对于大型代码库，首次处理可能需要较长时间
- 需确保有足够的内存用于模型加载和向量生成
- 建议在处理前备份原始代码库

## 扩展应用

- 代码相似度检测
- 代码智能推荐
- 大型代码库知识图谱构建
- 代码文档自动生成