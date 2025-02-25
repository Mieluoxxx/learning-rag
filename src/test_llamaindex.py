"""
Author: Morgan Woods weiyiding0@gmail.com
Date: 2025-02-25 22:02:33
LastEditors: Morgan Woods weiyiding0@gmail.com
LastEditTime: 2025-02-25 22:09:21
FilePath: /learning-rag/src/llama-index.py
Description: 
"""

"""
Author: Morgan Woods weiyiding0@gmail.com
Date: 2025-02-25 22:02:33
LastEditors: Morgan Woods weiyiding0@gmail.com
LastEditTime: 2025-02-25 22:02:48
FilePath: /learning-rag/src/llama-index.py
Description: 
"""
import chromadb

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding


# 设置模型
Settings.llm = Ollama(model="qwen2.5:3b")
Settings.embed_model = OllamaEmbedding(model_name="milkey/dmeta-embedding-zh:f16")

# 加载与读取文档
reader = SimpleDirectoryReader(
    input_files=["data/deepseek.txt", 
                 "data/enriebot.txt"]
)
documents = reader.load_data()

# 分割文档
node_parser = SentenceSplitter(chunk_size=500, chunk_overlap=20)
nodes = node_parser.get_nodes_from_documents(documents, show_progress=False)

# 准备向量存储
chroma = chromadb.HttpClient(host="localhost", port=8000)
chroma.delete_collection(name="ragdb")
collection = chroma.get_or_create_collection(
    name="ragdb", metadata={"hnsw:space": "cosine"}
)
vector_store = ChromaVectorStore(chroma_collection=collection)

# 准备向量存储索引
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex(nodes, storage_context=storage_context)

# 构造查询引擎
query_engine = index.as_query_engine()
while True:
    user_input = input("问题：")
    if user_input.lower() == "exit":
        break
    response = query_engine.query(user_input)
    print("AI 助手：", response.response)
