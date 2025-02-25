"""
Author: Morgan Woods weiyiding0@gmail.com
Date: 2025-02-25 22:18:37
LastEditors: Morgan Woods weiyiding0@gmail.com
LastEditTime: 2025-02-25 22:23:29
FilePath: /learning-rag/src/test_langchain.py
Description: 
"""

import chromadb
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain import hub
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader

# 模型
llm = OllamaLLM(model="qwen2.5:7b")
embed_model = OllamaEmbeddings(model="milkey/dmeta-embedding-zh:f16")

# 加载与读取文档
loader = DirectoryLoader(
    "data/", glob="*.txt", exclude="*tips*.txt", loader_cls=TextLoader
)
documents = loader.load()

# 分割文档
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
splits = text_splitter.split_documents(documents)

# 准备向量存储
chroma = chromadb.HttpClient(host="localhost", port=8000)
chroma.delete_collection(name="ragdb")

collection = chroma.get_or_create_collection(
    name="ragdb", metadata={"hnsw:space": "cosine"}
)
db = Chroma(client=chroma, collection_name="ragdb", embedding_function=embed_model)

# 存储到向量库中，构造索引
db.add_documents(splits)

# 使用检索器
retriever = db.as_retriever()

# 构造一个 RAG“链”（使用 LangChain 框架特有的组件与表达语言）
prompt = hub.pull("rlm/rag-prompt")
rag_chain = (
    {
        "context": retriever
        | (lambda docs: "\n\n".join(doc.page_content for doc in docs)),
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)
while True:
    user_input = input("问题：")
    if user_input.lower() == "exit":
        break
    response = rag_chain.invoke(user_input)
    print("AI 助手：", response)
