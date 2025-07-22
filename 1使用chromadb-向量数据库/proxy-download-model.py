from chromadb import HttpClient
import os
from sentence_transformers import SentenceTransformer

# 设置代理
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:1087'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:1087'

# 1. 加载 BAAI 中文模型（首次运行会自动下载）
# model = SentenceTransformer("/Users/yxy/work/ai/model/bge-small-zh-v1.5")
model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

# 将模型保存到你的自定义目录（覆盖旧文件）
model.save("/Users/yxy/work/ai/model-new/bge-small-zh-v1.5")
