from chromadb import HttpClient

# 连接到 Chroma
client = HttpClient(host="localhost", port=8000)

# 创建集合
# collection = client.create_collection(name="docs")

# # 添加数据
# collection.add(
#     documents=["苹果是水果", "汽车是交通工具"],
#     metadatas=[{"type": "水果"}, {"type": "交通工具"}],
#     ids=["id1", "id2"]
# )

collection = client.get_collection(name="docs")

# 查询相似结果
results = collection.query(
    query_texts=["水果"],
    n_results=1
)

print(results["documents"][0])  # 输出: ['苹果是水果']
