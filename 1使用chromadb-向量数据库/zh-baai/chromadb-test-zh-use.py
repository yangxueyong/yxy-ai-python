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

    def name(self) -> str:
        return "bge-small-zh-v1.5"

    def __eq__(self, other):
        if not isinstance(other, BGEEmbeddingFunction):
            return False
        return self.name() == other.name()


# 创建集合
collection = client.get_collection(
    name="docs_for6_baai",
    embedding_function=BGEEmbeddingFunction(model)
)

# 5. 查询
results = collection.query(query_texts=["火车"], n_results=1)
print(results["documents"][0])
