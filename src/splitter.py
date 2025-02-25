'''
Author: Morgan Woods weiyiding0@gmail.com
Date: 2025-02-25 20:22:53
LastEditors: Morgan Woods weiyiding0@gmail.com
LastEditTime: 2025-02-25 21:08:29
FilePath: /learning-rag/src/splitter.py
Description: 
'''
import re
from typing import List


def split_text_by_sentence(
    source_text: str, sentences_per_chunk: int, overlap: int
) -> List[str]:
    """
    将文档分割为多个知识块，每个块包含指定数量的句子，并允许一定的重叠。

    :param source_text: 源文本
    :param sentences_per_chunk: 每个块包含的句子数
    :param overlap: 句子重叠的数量
    :return: 分割后的文本块列表
    """
    if sentences_per_chunk < 2:
        raise ValueError("每个块至少需要包含 2 个句子！")
    if overlap < 0 or overlap >= sentences_per_chunk:
        raise ValueError("overlap 参数必须大于等于 0，且小于 sentences_per_chunk")

    # 使用正则表达式分割句子
    sentences = re.split(r"(?<=[。！？])\s+", source_text)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    
    if not sentences:
        print("没有可分割的句子。")
        return []

    chunks = []
    i = 0
    while i < len(sentences):
        end = min(i + sentences_per_chunk, len(sentences))
        chunk_sentences = sentences[i:end]
        
        # 处理重叠部分
        if overlap > 0 and i > 0:
            overlap_sentences = sentences[i - overlap:i]
            chunk_sentences = overlap_sentences + chunk_sentences
        
        chunk = " ".join(chunk_sentences)
        chunks.append(chunk.strip())
        i += sentences_per_chunk

    return chunks
