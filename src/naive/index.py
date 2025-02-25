"""
Author: Morgan Woods weiyiding0@gmail.com
Date: 2025-02-25 20:33:52
LastEditors: Morgan Woods weiyiding0@gmail.com
LastEditTime: 2025-02-25 20:41:18
FilePath: /learning-rag/src/emb.py
Description: 
"""

import ollama, chromadb

# 自定义模块
from load import loadtext, getconfig
from splitter import split_text_by_sentence

# 向量模型
embedmodel = getconfig()["embedmodel"]

# 向量库
chroma = chromadb.HttpClient(host="localhost", port=8000)
chroma.delete_collection(name="ragdb")
collection = chroma.get_or_create_collection(name="ragdb")

# 读取文档列表，依次处理
with open("docs.txt") as f:
    lines = f.readlines()
    for filename in lines:
        # 加载文档内容
        print(filename)
        text = loadtext(filename)
        print(text)
        print("----")
        # 把文档分割成知识块
        chunks = split_text_by_sentence(
            source_text=text, sentences_per_chunk=8, overlap=0
        )

        # 对知识块依次处理
        for index, chunk in enumerate(chunks):
            # 借助基于 Ollama 部署的本地嵌入模型生成向量
            embed = ollama.embeddings(model=embedmodel, prompt=chunk)["embedding"]
            # 存储到向量库 Chroma 中，注意这里的参数
            collection.add(
                [filename + str(index)],
                [embed],
                documents=[chunk],
                metadatas={"source": filename},
            )

if __name__ == "__main__":
    while True:
        query = input("Enter your query: ")
        if query.lower() == 'quit':
            break
        else:
            # 从向量库 Chroma 中查询与向量相似的知识块
            results = collection.query(query_embeddings=[ollama.embeddings(model=embedmodel, prompt=query)['embedding']], n_results=3)
            # 打印文档内容（Chunk）
            for result in results["documents"][0]:
                print("----------------------------------------------------") 
                print(result)