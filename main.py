# encoding:utf-8

"""
源代码嵌入向量生成与检索工具

本模块提供了源代码文件处理、向量化和检索功能，支持:
1. 读取指定目录下的代码文件
2. 生成代码的嵌入向量并存储到Milvus向量数据库
3. 通过语义搜索快速查找相关代码片段
4. 使用Redis缓存文件MD5值，避免重复处理未变更文件
"""

import redis
import os
import sys
import glob
import ast
import hashlib
from transformers import AutoTokenizer
from pymilvus import MilvusClient, DataType, FieldSchema, CollectionSchema, model
from loguru import logger

# 配置日志记录
# 移除默认处理器
logger.remove()

# 添加无彩色的控制台输出，添加毫秒级时间显示
logger.add(
    sys.stdout,
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    level="INFO",
    colorize=False,
)

# 添加文件日志，使用相同的格式
logger.add(
    "logs/app_{time:YYYY-MM-DD}.log",
    rotation="00:00",
    retention="30 days",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    level="INFO",
    encoding="utf-8",
    enqueue=True,
)


class SourceCodeEmbedding:
    """
    源代码嵌入向量处理类

    该类用于处理代码文件，生成嵌入向量，并提供向量检索功能。
    支持增量更新，仅处理已变更的文件。
    """

    def __init__(self, embeddings_model, directory_path, file_extension):
        """
        初始化源代码嵌入处理器

        Args:
            embeddings_model (str): 嵌入模型名称，例如 'BAAI/bge-m3'
            directory_path (str): 要处理的源代码目录路径
            file_extension (str): 要处理的文件扩展名，例如 'py'
        """
        self.embeddings_model = embeddings_model
        self.tokenizer = AutoTokenizer.from_pretrained(embeddings_model)
        # 设置最大token数，预留1000个token作为安全边界
        # self.MAX_TOKENS = self.tokenizer.model_max_length - 1000
        self.MAX_TOKENS = 2000
        self.directory_path = directory_path
        self.file_extension = file_extension
        self.COLLECTION_NAME = "devops_demo"
        self.client = MilvusClient("milvus_demo.db")
        self.REDIS_PREFIX = "file_md5:"  # Redis键前缀，用于缓存文件MD5
        self.data = []  # 存储待插入Milvus的数据
        self.update_file_list = []  # 记录已更新的文件列表
        self.index_field = "vector"

        # 初始化Milvus和Redis连接
        self._init_milvus()
        self._init_redis()

        # 初始化嵌入模型
        self._embedding_function = model.dense.SentenceTransformerEmbeddingFunction(
            model_name=self.embeddings_model,
            device="cpu",
            trust_remote_code=True,
        )

    def _init_redis(self):
        """
        初始化Redis连接

        连接到Redis服务器用于缓存文件MD5值，以支持增量更新。
        如果连接失败，程序将退出。
        """
        try:
            self.redis_client = redis.Redis(
                host="172.16.11.105", port=16379, db=0, decode_responses=True
            )
            self.redis_client.ping()
            logger.info("Redis连接成功")
        except redis.ConnectionError as e:
            logger.error(f"Redis连接失败: {e}")
            logger.error("程序退出")
            sys.exit(1)

    def _init_milvus(self):
        """
        初始化Milvus向量数据库

        创建集合和索引用于存储和检索代码嵌入向量。
        如果集合已存在，则跳过创建步骤。
        """
        # 定义集合字段结构
        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.VARCHAR,
                is_primary=True,
                auto_id=True,
                max_length=100,
            ),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
            FieldSchema(name="path", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=10000),
        ]
        schema = CollectionSchema(
            fields, "SNOMED-CT Concepts", enable_dynamic_field=True
        )

        # 如果集合不存在，则创建
        if not self.client.has_collection(self.COLLECTION_NAME):
            self.client.create_collection(
                collection_name=self.COLLECTION_NAME, schema=schema, dimension=1024
            )
            logger.info("创建集合完成")

        # 如果索引不存在，则创建
        indexes = self.client.list_indexes(collection_name=self.COLLECTION_NAME)

        if self.index_field not in indexes:
            
            # 创建索引
            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name=self.index_field,  # 指定要为哪个字段创建索引，这里是向量字段
                index_type="AUTOINDEX",  # 使用自动索引类型，Milvus会根据数据特性选择最佳索引
                metric_type="COSINE",  # 使用余弦相似度作为向量相似度度量方式
                params={
                    "nlist": 1024
                },  # 索引参数：nlist表示聚类中心的数量，值越大检索精度越高但速度越慢
            )

            self.client.create_index(
                collection_name=self.COLLECTION_NAME, index_params=index_params
            )
            logger.info("创建索引完成")

    def count_tokens(self, text):
        """
        计算文本的token数量

        Args:
            text (str): 要计算token数量的文本

        Returns:
            int: token数量
        """
        return len(self.tokenizer.encode(text, truncation=False))

    def _get_file_md5(self, file_path):
        """
        计算文件内容的MD5哈希值

        Args:
            file_path (str): 文件路径

        Returns:
            str: 文件内容的MD5哈希值
        """
        with open(file_path, "r", encoding="utf-8") as file:
            return hashlib.md5(file.read().encode("utf-8")).hexdigest()

    def _extract_top_level_nodes(self, code_str):
        """
        从Python代码中提取顶层节点（类和函数）

        当文件超过token限制时，使用AST将代码分解为更小的块，
        以便可以单独处理每个函数或类。

        Args:
            code_str (str): Python源代码字符串

        Returns:
            list: 提取出的顶层节点代码片段列表
        """
        try:
            tree = ast.parse(code_str)
            chunks = []

            for node in tree.body:
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    source = ast.get_source_segment(code_str, node)
                    if source:
                        chunks.append(source)

            return chunks
        except SyntaxError as e:
            logger.error(f"AST解析错误: {e}")
            return []

    def _list_all_files(self):
        """
        获取目录中所有指定扩展名的文件

        Returns:
            list: 符合条件的文件路径列表
        """
        return glob.glob(
            os.path.join(self.directory_path, f"**/*{self.file_extension}"),
            recursive=True,
        )

    def _embedding_text(self, text):
        """
        生成文本的嵌入向量

        Args:
            text (str): 要嵌入的文本

        Returns:
            list: 嵌入向量，如果发生错误则返回空列表
        """
        embeddings = []
        try:
            # 生成嵌入向量
            embeddings = self._embedding_function.encode_documents([text])

            # 修正向量格式：取出第一个文档的嵌入向量，确保它是单层浮点数列表
            if isinstance(embeddings, list) and len(embeddings) > 0:
                embeddings = (
                    embeddings[0].tolist()
                    if hasattr(embeddings[0], "tolist")
                    else embeddings[0]
                )
                # 确保embeddings是简单的浮点数列表
                if not isinstance(embeddings, list):
                    embeddings = list(embeddings)
                    logger.warning("生成的嵌入向量格式不正确")
        except Exception as e:
            logger.error(f"生成嵌入向量时出错: {str(e)}")

        return embeddings

    def _insert_data_to_milvus(self):
        """
        将数据插入Milvus数据库

        将处理好的数据批量插入到Milvus集合中。
        """
        try:
            self.client.upsert(self.COLLECTION_NAME, self.data)
            logger.info(f"插入数据到Milvus完成, 插入数据数量: {len(self.data)}")
        except Exception as e:
            logger.error(f"插入数据到Milvus时出错: {str(e)}")

    def run(self):
        """
        执行源代码处理和向量化

        主处理流程，包括：
        1. 列出所有匹配的文件
        2. 检查文件是否已更新（通过MD5比较）
        3. 处理更新的文件，生成嵌入向量
        4. 对于超过token限制的Python文件，使用AST提取顶层节点
        5. 更新Redis缓存和Milvus向量库
        """
        files = self._list_all_files()
        logger.info(f"找到 {len(files)} 个文件")

        # 处理每个文件
        for file in files:
            file_md5 = self._get_file_md5(file)
            # 检查文件是否已更新，如果MD5相同则跳过
            if self.redis_client.get(f"{self.REDIS_PREFIX}{file}") == file_md5:
                logger.info(f"文件 {file} 未变更，跳过处理")
                continue

            with open(file, "r", encoding="utf-8") as f:
                content = f.read()
                tokens_count = self.count_tokens(content)
                logger.info(f"文件 {file} 的tokens数量: {tokens_count}")
                self.update_file_list.append(file)

                # 跳过空文件
                if len(content) == 0:
                    logger.warning(f"文件 {file} 为空，跳过处理")
                    continue

                # 处理超过token限制的文件
                if tokens_count > self.MAX_TOKENS:
                    logger.warning(
                        f"文件 {file} 的tokens数量超过限制: {tokens_count}, 使用AST提取代码块"
                    )
                    chunks = self._extract_top_level_nodes(content)

                    length = len(chunks)
                    file_chunk = 0
                    # 处理每个代码块
                    for chunk in chunks:
                        chunk_tokens = self.count_tokens(chunk)
                        if chunk_tokens > self.MAX_TOKENS:
                            continue
                        file_chunk += 1
                        logger.info(
                            f"处理文件 {file} 的代码块, 当前代码块: {file_chunk}, 总代码块: {length}"
                        )

                        embeddings = self._embedding_text(chunk)

                        if embeddings:
                            data = {
                                "id": hashlib.md5(chunk.encode()).hexdigest(),
                                "path": file,
                                "content": chunk,
                                "vector": embeddings,
                            }
                            self.data.append(data)
                    # 更新文件MD5到Redis
                    self.redis_client.set(f"{self.REDIS_PREFIX}{file}", file_md5)
                    continue

                # 处理未超过token限制的文件
                logger.info(f"处理文件 {file} 的全部代码")
                self.redis_client.set(f"{self.REDIS_PREFIX}{file}", file_md5)
                embeddings = self._embedding_text(content)
                if embeddings:
                    data = {
                        "id": hashlib.md5(content.encode()).hexdigest(),
                        "path": file,
                        "content": content,
                        "vector": embeddings,
                    }
                    self.data.append(data)

        # 如果有更新的文件，先删除旧数据
        if self.update_file_list:
            logger.info(f"更新了 {len(self.update_file_list)} 个文件")
            for file in self.update_file_list:
                self.delete_milvus(file, delete_redis=False)

        # 如果有数据需要插入，执行插入操作
        if self.data:
            self._insert_data_to_milvus()

    def search_milvus(self, query):
        """
        在Milvus中搜索相似的代码

        Args:
            query (str): 查询文本

        Returns:
            list: 搜索结果列表
        """
        logger.info(f"执行查询: {query}")
        try:
            # 生成查询文本的嵌入向量
            query_embeddings = self._embedding_function.encode_documents([query])[
                0
            ].tolist()

            # 确保向量中所有值都是浮点数
            query_embeddings = [float(val) for val in query_embeddings]

            # 执行向量搜索
            results = self.client.search(
                collection_name=self.COLLECTION_NAME,
                data=[query_embeddings],
                output_fields=["path", "content"],
                limit=5,  # 返回前5个最相似的结果
            )
            logger.info(f"查询完成，返回 {len(results)} 条结果")
            return results
        except Exception as e:
            logger.error(f"搜索Milvus时出错: {str(e)}")
            import traceback

            logger.error(traceback.format_exc())
            return []

    def query_milvus(self, path):
        """
        通过路径查询Milvus中的数据

        Args:
            path (str): 文件路径或路径片段

        Returns:
            list: 查询结果列表
        """
        logger.info(f"查询路径: {path}")
        try:
            results = self.client.query(
                collection_name=self.COLLECTION_NAME,
                output_fields=["path", "content"],
                filter=f"path like '%{path}%'",
            )
            logger.info(f"查询完成，返回 {len(results)} 条结果")
            return results
        except Exception as e:
            logger.error(f"查询Milvus时出错: {str(e)}")
            return []

    def delete_milvus(self, path, delete_redis=True):
        """
        删除Milvus中的数据

        Args:
            path (str): 要删除的文件路径或路径片段
            delete_redis (bool): 是否同时删除Redis中的缓存

        Returns:
            int: 删除的记录数
        """
        logger.info(f"删除路径: {path}")
        try:
            # 先查询匹配的记录
            results = self.query_milvus(path)

            # 删除每条记录
            for result in results:
                logger.info("删除Milvus中的数据")
                self.client.delete(
                    collection_name=self.COLLECTION_NAME,
                    filter=f"id == '{result['id']}'",
                )
                redis_path = result["path"]
                # 如果需要，同时删除Redis缓存
                if delete_redis:
                    logger.info(f"删除Redis中的数据: {self.REDIS_PREFIX}{redis_path}")
                    self.redis_client.delete(f"{self.REDIS_PREFIX}{redis_path}")

            return len(results)
        except Exception as e:
            logger.error(f"删除Milvus数据时出错: {str(e)}")
            return 0


def main():
    """主函数，演示如何使用SourceCodeEmbedding类"""
    directory_path = "demo_src"
    file_extension = "py"
    embedding_model = "BAAI/bge-m3"

    logger.info(f"开始处理 {directory_path} 目录下的 {file_extension} 文件")
    source_code_embedding = SourceCodeEmbedding(
        embedding_model, directory_path, file_extension
    )
    # 向量化源代码
    source_code_embedding.run()

    # 搜索示例
    logger.info("执行搜索示例")
    results = source_code_embedding.search_milvus("REDIS的默认配置")

    # 处理并显示搜索结果
    if results and len(results) > 0:
        # 检查返回的结果结构
        logger.debug(f"结果类型: {type(results)}")

        # 处理第一个查询的结果
        query_results = results[0]

        if query_results:
            logger.info(f"找到 {len(query_results)} 条匹配结果")

            # 显示每条结果的信息
            for i, hit in enumerate(query_results):
                logger.info(f"结果 {i+1}: 相似度: {hit['distance']}, 路径: {hit['entity'].get('path')}")
                if 'content' in hit['entity']:
                    content = hit['entity']['content']
                    logger.info(f"内容摘要: {content[:100]}..." if content else "无内容")
        else:
            logger.info("未找到匹配结果")
    else:
        logger.info("搜索没有返回结果")

    # 以下是其他操作的示例，默认注释掉
    # 查询示例
    # logger.info("执行查询示例")
    # res = source_code_embedding.query_milvus("devops/demo.py")
    # print(res)

    # 删除示例
    # logger.info("执行删除示例")
    # res = source_code_embedding.delete_milvus("devops/demo.py")
    # print(res)


if __name__ == "__main__":
    main()

