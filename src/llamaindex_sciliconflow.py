'''
Author: Morgan Woods weiyiding0@gmail.com
Date: 2025-02-26 10:05:38
LastEditors: Morgan Woods weiyiding0@gmail.com
LastEditTime: 2025-02-26 10:10:35
FilePath: /learning-rag/src/llamaindex_sciliconflow.py
Description: 
'''
import chromadb
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore

from llama_index.embeddings.siliconflow import SiliconFlowEmbedding
from llama_index.llms.siliconflow import SiliconFlow

Settings.llm = SiliconFlow(
    model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    api_key="sk-grhadewuldjevwtpelpfjwgxzyuwqacznfyibbhsjhkvigiy",
)
Settings.embed_model = SiliconFlowEmbedding(
    model="BAAI/bge-m3",
    api_key="sk-grhadewuldjevwtpelpfjwgxzyuwqacznfyibbhsjhkvigiy",
)

# 加载与读取文档
reader = SimpleDirectoryReader(
    input_files=[
        "data/deepseek.txt",
        "data/enriebot.txt"
    ]
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
