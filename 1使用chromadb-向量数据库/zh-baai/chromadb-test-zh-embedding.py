from chromadb import HttpClient
from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

# 1. 加载中文模型
model = SentenceTransformer("/Users/yxy/work/ai/model-new/bge-small-zh-v1.5")

# 2. 连接到 ChromaDB
client = HttpClient(host="localhost", port=8000)


# 3. 创建自定义嵌入函数
class BGEEmbeddingFunction:
    def __init__(self, model):
        self._model = model

    def __call__(self, input: List[str]) -> List[List[float]]:
        # 返回 numpy 数组，让 ChromaDB 自己处理转换
        embeddings = self._model.encode(input, normalize_embeddings=True)
        return embeddings  # 直接返回 numpy 数组


# 创建集合
collection = client.create_collection(
    name="docs_for6_baai",
    embedding_function=BGEEmbeddingFunction(model)
)

# 4. 添加数据
collection.add(
    documents=["苹果是一种常见的水果", "高铁是中国自主研发的交通工具"],
    metadatas=[{"type": "水果"}, {"type": "交通工具"}],
    ids=["id1", "id2"]
)

# 5. 查询
results = collection.query(query_texts=["火车"], n_results=1)
print(results["documents"][0])
