#!/usr/bin/env python3
"""
使用 BM25S 检索器和 vLLM 生成器的 RAG 流水线脚本

支持两种模式：
1. 简单 RAG：检索一次后生成回答
2. 逐句检索 RAG：每生成一句话后再次检索

用法：
    python rag_pipeline.py --mode simple
    python rag_pipeline.py --mode fix-sentence --corpus BeIR/hotpotqa --model meta-llama/Llama-3.2-1B
"""

import argparse
import json
import logging
import os
import pickle
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import bm25s
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from run_bm25_loader import run_bm25_loader
from bm25_loader_xrt import *
import pyxrt

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class LatencyTracker:
    """记录并报告延迟指标。"""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
    
    def start(self, name: str):
        """开始对指定指标计时。"""
        self.start_times[name] = time.perf_counter()
    
    def stop(self, name: str) -> float:
        """停止计时并记录指标。"""
        if name not in self.start_times:
            logger.warning(f"Timer '{name}' was never started")
            return 0.0
        elapsed = time.perf_counter() - self.start_times[name]
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(elapsed)
        del self.start_times[name]
        return elapsed
    
    def get_summary(self) -> dict:
        """获取所有指标的统计摘要。"""
        summary = {}
        for name, times in self.metrics.items():
            summary[name] = {
                'count': len(times),
                'total_ms': sum(times) * 1000,
                'avg_ms': (sum(times) / len(times)) * 1000 if times else 0,
                'min_ms': min(times) * 1000 if times else 0,
                'max_ms': max(times) * 1000 if times else 0,
            }
        return summary
    
    def print_summary(self):
        """按格式打印所有指标的统计摘要。"""
        summary = self.get_summary()
        logger.info("=" * 60)
        logger.info("LATENCY METRICS SUMMARY")
        logger.info("=" * 60)
        for name, stats in summary.items():
            logger.info(f"{name}:")
            logger.info(f"  Count: {stats['count']}")
            logger.info(f"  Total: {stats['total_ms']:.2f} ms")
            logger.info(f"  Avg: {stats['avg_ms']:.2f} ms")
            logger.info(f"  Min: {stats['min_ms']:.2f} ms")
            logger.info(f"  Max: {stats['max_ms']:.2f} ms")
        logger.info("=" * 60)


class BM25Retriever:
    """使用自定义分词器的 BM25S 文档检索器。"""
    
    def __init__(self, corpus_name: str, cache_dir: str = "./cache", 
                 retriever_tokenizer_name: str = "EleutherAI/gpt-j-6b", device: str = 'cuda'):
        self.corpus_name = corpus_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.retriever = None
        self.corpus_texts = None
        self.corpus_ids = None
        self.latency = LatencyTracker()
        self.retriever_tokenizer_name = retriever_tokenizer_name
        self.hf_tokenizer = None  # 用于 BM25 的 HuggingFace 分词器
        self.vocab_dict = {}
        self.device = device
        self.fpga_setup = None
    
    def _get_cache_path(self) -> Path:
        """获取已分词语料的缓存路径。"""
        safe_name = self.corpus_name.replace("/", "_").replace("\\", "_")
        tokenizer_suffix = self.retriever_tokenizer_name.replace("/", "_").replace("\\", "_")
        return self.cache_dir / f"{safe_name}_bm25s_{tokenizer_suffix}"
    
    def _load_corpus(self) -> Tuple[List[str], List[str]]:
        """从 HuggingFace Datasets 加载语料。"""
        logger.info(f"Loading corpus: {self.corpus_name}")
        self.latency.start("corpus_loading")
        
        try:
            # 加载 BeIR 数据集的 corpus 子集
            if "BeIR" in self.corpus_name or "beir" in self.corpus_name.lower():
                dataset_name = self.corpus_name
                dataset = load_dataset(dataset_name, "corpus", trust_remote_code=True)
                
                # BeIR 数据集通常包含 'corpus' 划分
                if "corpus" in dataset:
                    corpus_data = dataset["corpus"]
                elif "train" in dataset:
                    corpus_data = dataset["train"]
                else:
                    # 使用第一个可用的数据划分
                    split_name = list(dataset.keys())[0]
                    corpus_data = dataset[split_name]
                
                # 提取正文和 ID
                texts = []
                ids = []
                for item in corpus_data:
                    # 如果有标题和正文，则将它们合并
                    text_parts = []
                    if "title" in item and item["title"]:
                        text_parts.append(item["title"])
                    if "text" in item and item["text"]:
                        text_parts.append(item["text"])
                    
                    if text_parts:
                        texts.append(" ".join(text_parts))
                        ids.append(item.get("_id", str(len(ids))))
            else:
                # 加载通用数据集
                dataset = load_dataset(self.corpus_name, trust_remote_code=True)
                split_name = list(dataset.keys())[0]
                corpus_data = dataset[split_name]
                
                texts = []
                ids = []
                for i, item in enumerate(corpus_data):
                    # 尝试常见的正文字段名
                    text = item.get("text") or item.get("content") or item.get("document") or str(item)
                    texts.append(text)
                    ids.append(item.get("id", str(i)))
            
            elapsed = self.latency.stop("corpus_loading")
            logger.info(f"Loaded {len(texts)} documents in {elapsed*1000:.2f} ms")
            
            return texts, ids
            
        except Exception as e:
            self.latency.stop("corpus_loading")
            logger.error(f"Failed to load corpus: {e}")
            raise
    
    def _tokenize_with_hf(self, texts: List[str], batch_size: int = 1000) -> List[List[str]]:
        """使用 HuggingFace 分词器（默认为 GPT-J）对文本分词。
        
        采用批量分词以提高效率。将 token ID 转为字符串返回，
        BM25S 会将这些字符串视为词表词项。这样比逐个将 token
        解码回对应文本更快。
        
        参数：
            texts：待分词的文本列表
            batch_size：每批处理的文本数量
            
        返回：
            token 列表组成的列表；token ID 以字符串表示，以兼容 BM25S。
        """
        if self.hf_tokenizer is None:
            logger.info(f"Loading retriever tokenizer: {self.retriever_tokenizer_name}")
            self.hf_tokenizer = AutoTokenizer.from_pretrained(
                self.retriever_tokenizer_name,
                trust_remote_code=True
            )
            # 确保分词器具有填充 token，以支持批处理
            if self.hf_tokenizer.pad_token is None:
                self.hf_tokenizer.pad_token = self.hf_tokenizer.eos_token
        
        tokenized = []
        num_texts = len(texts)
        
        # 分批处理以提高效率
        for start_idx in range(0, num_texts, batch_size):
            end_idx = min(start_idx + batch_size, num_texts)
            batch_texts = texts[start_idx:end_idx]
            
            # 批量分词，比逐条分词更快
            batch_result = self.hf_tokenizer(
                batch_texts,
                add_special_tokens=False,
                return_attention_mask=False,
                padding=False,  # 不进行填充，保留可变长度
                truncation=False,
            )
            
            # 将 token ID 转为字符串，供 BM25S 使用
            # BM25S 将每个不同的字符串视为一个词表词项
            for token_ids in batch_result['input_ids']:
                tokens = [str(tid) for tid in token_ids]
                tokenized.append(tokens)
            
            if end_idx % 10000 == 0 or end_idx == num_texts:
                logger.info(f"Tokenized {end_idx}/{num_texts} documents")
        
        return tokenized
    
    def initialize(self):
        """初始化检索器；如果存在缓存，则从缓存加载。"""
        cache_path = self._get_cache_path()

        if self.device == 'hybrid':
            # 初始化 FPGA
            fpga_setup = fpga_retriever_setup(
                bitstream="../indexer_bm25.xclbin",
                export_dir="./export"
            )
            if fpga_setup is None:
                raise RuntimeError("Failed to set up FPGA retriever")
            self.fpga_setup = fpga_setup
        
        if cache_path.exists():
            logger.info(f"Loading tokenized corpus from cache: {cache_path}")
            self.latency.start("cache_loading")
            
            try:
                # 只加载 BM25 索引，不加载正文；corpus_texts 由本类单独管理
                # 避免 bm25s 在 retrieve() 中尝试直接返回文档
                self.retriever = bm25s.BM25.load(cache_path, load_corpus=False)
                
                # 从单独保存的元数据文件加载文档正文和 ID
                corpus_meta_path = cache_path / "corpus_meta.pkl"
                if corpus_meta_path.exists():
                    with open(corpus_meta_path, "rb") as f:
                        meta = pickle.load(f)
                        self.corpus_texts = meta["texts"]
                        self.corpus_ids = meta["ids"]
                else:
                    raise FileNotFoundError("corpus_meta.pkl not found in cache")
                
                # 检查 corpus_texts 是否已正确加载
                if self.corpus_texts is None or len(self.corpus_texts) == 0:
                    raise ValueError("Corpus texts are empty after loading from cache")
                
                # 如果存在词表缓存，则加载词表映射
                vocab_path = cache_path / "vocab.index.json"
                if vocab_path.exists():
                    with open(vocab_path, "r") as f:
                        self.vocab_dict = json.load(f)
                else:
                    raise FileNotFoundError("vocab.index.json not found in cache")
                
                elapsed = self.latency.stop("cache_loading")
                logger.info(f"Loaded cached retriever in {elapsed*1000:.2f} ms")
                logger.info(f"Corpus size: {len(self.corpus_texts)} documents")
                return
                
            except Exception as e:
                self.latency.stop("cache_loading")
                logger.warning(f"Failed to load cache, rebuilding: {e}")
        
        # 加载语料并分词
        self.corpus_texts, self.corpus_ids = self._load_corpus()
        
        logger.info(f"Tokenizing corpus with {self.retriever_tokenizer_name} tokenizer...")
        self.latency.start("tokenization")
        
        # 使用 HuggingFace 分词器（GPT-J）对语料分词
        corpus_tokens = self._tokenize_with_hf(self.corpus_texts)
        
        elapsed = self.latency.stop("tokenization")
        logger.info(f"Tokenization completed in {elapsed*1000:.2f} ms")
        
        # 创建 BM25 检索器
        logger.info("Building BM25 index...")
        self.latency.start("indexing")
        
        self.retriever = bm25s.BM25()
        self.retriever.index(corpus_tokens)
        
        elapsed = self.latency.stop("indexing")
        logger.info(f"Indexing completed in {elapsed*1000:.2f} ms")
        
        # 保存到缓存
        logger.info(f"Saving tokenized corpus to cache: {cache_path}")
        self.latency.start("cache_saving")
        
        # 保存 BM25 索引，不包含正文；正文单独保存在 corpus_meta.pkl 中
        self.retriever.save(cache_path)
        
        # 保存语料元数据（原始正文和 ID）
        corpus_meta_path = cache_path / "corpus_meta.pkl"
        with open(corpus_meta_path, "wb") as f:
            pickle.dump({"texts": self.corpus_texts, "ids": self.corpus_ids}, f)
        
        elapsed = self.latency.stop("cache_saving")
        logger.info(f"Cache saved in {elapsed*1000:.2f} ms")
    
    def get_documents(self, doc_ids: List[int]) -> List[Tuple[str, float]]:
        """根据文档下标获取正文。"""
        if self.corpus_texts is None:
            raise RuntimeError("Corpus texts not loaded. Call initialize() first.")
        
        docs = []
        for idx in doc_ids:
            if 0 <= idx < len(self.corpus_texts):
                docs.append((self.corpus_texts[idx], 0))
            else:
                logger.warning(f"Invalid document index requested: {idx}")
                docs.append(("", 0))
        return docs
    
    def retrieve(self, query: str, k: int = 64) -> List[Tuple[str, float]]:
        """检索与查询最相关的 Top-K 文档。"""
        if self.retriever is None:
            raise RuntimeError("Retriever not initialized. Call initialize() first.")
        
        if self.corpus_texts is None or len(self.corpus_texts) == 0:
            raise RuntimeError("Corpus texts not loaded. Re-initialize the retriever.")
        
        # 处理空查询或很短的查询
        if not query or len(query.strip()) == 0:
            logger.warning("Empty query provided, returning empty results")
            return []
        
        query_preview = query[:100] if len(query) > 100 else query
        logger.info(f"Retrieving top-{k} documents for query: '{query_preview}...'")
        
        # 使用 HuggingFace 分词器（GPT-J）对查询分词
        query_tokens = self._tokenize_with_hf([query])
        logger.info(f"Tokenized query to {len(query_tokens[0])} tokens")
        logger.info(f"query tokens ids: {query_tokens[0]}")

        if self.device == 'hybrid':

            #self.latency.start("retrieval")
            logger.info("Launching FPGA retriever...")

            query_ids = [str(self.vocab_dict.get(str(tid), -1)) for tid in query_tokens[0]]
            query_ids_str = ",".join(query_ids)

            kernel, L, L_doc_total, bo_query_bitmap, bo_df_buffer, bo_inst_mem, bo_doc_mem_0, bo_doc_mem_1, bo_doc_mem_2, bo_doc_mem_3, bo_topk_id, buf_topk = self.fpga_setup
            
            topk_ids, kernel_latency = fpga_retriver_launch(kernel, L, L_doc_total, bo_query_bitmap, bo_df_buffer, bo_inst_mem, bo_doc_mem_0, bo_doc_mem_1, bo_doc_mem_2, bo_doc_mem_3, bo_topk_id, buf_topk,
                                                            query_ids_str)
            indices = topk_ids[:-3]
            
            #elapsed = self.latency.stop("retrieval")
            elapsed = kernel_latency
            logger.info(f"Retrieval completed in {elapsed:.2f} ms")
            logger.info(f"top indices: {indices}")
            
            # 获取文档正文，并检查下标有效性
            # results 和 scores 是形状为 (n_queries, k) 的 NumPy 数组
            retrieved_docs = []
            try:
                # results[0] 给出第一个（也是唯一一个）查询的文档下标
                for i in range(len(indices)):
                    idx = int(indices[i])
                    if 0 <= idx < len(self.corpus_texts):
                        retrieved_docs.append((self.corpus_texts[idx], 0))
                    else:
                        logger.warning(f"Invalid document index {idx}, skipping")
            except Exception as e:
                logger.error(f"Error processing retrieval results: {e}")
        else:
            self.latency.start("retrieval")
            
            # 当语料数量不足时调整 k
            effective_k = min(k, len(self.corpus_texts))
            if effective_k < k:
                logger.warning(f"Corpus size ({len(self.corpus_texts)}) is smaller than k ({k}), using k={effective_k}")
            
            if effective_k == 0:
                logger.warning("Effective k is 0, returning empty results")
                self.latency.stop("retrieval")
                return []
            
            # 执行检索，返回形状为 (n_queries, k) 的 NumPy 数组
            # 注意：这里不传入正文；bm25s 返回下标，再由本类映射到 corpus_texts
            results, scores = self.retriever.retrieve(query_tokens, k=effective_k)
            
            elapsed = self.latency.stop("retrieval") * 1000
            logger.info(f"Retrieval completed in {elapsed:.2f} ms")
            logger.info(f"top indices: {results[0]}")
            
            # 获取文档正文，并检查下标有效性
            # results 和 scores 是形状为 (n_queries, k) 的 NumPy 数组
            retrieved_docs = []
            try:
                # results[0] 给出第一个（也是唯一一个）查询的文档下标
                for i in range(len(results[0])):
                    idx = int(results[0][i])
                    score = float(scores[0][i])
                    if 0 <= idx < len(self.corpus_texts):
                        retrieved_docs.append((self.corpus_texts[idx], score))
                    else:
                        logger.warning(f"Invalid document index {idx}, skipping")
            except Exception as e:
                logger.error(f"Error processing retrieval results: {e}")
        
        logger.info(f"Retrieved {len(retrieved_docs)} documents")
        return retrieved_docs, elapsed


class RAGGenerator:
    """使用 vLLM、支持 RAG 的大语言模型生成器。"""
    
    def __init__(self, model_name: str, device: str = None, tensor_parallel_size: int = 1):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tensor_parallel_size = tensor_parallel_size
        self.llm = None
        self.tokenizer = None
        self.latency = LatencyTracker()
        
        # 逐句检索模式使用的句末标记
        self.sentence_end_chars = {'.', '!', '?', '\n'}
    
    def initialize(self):
        """使用 vLLM 加载模型。"""
        logger.info(f"Loading model with vLLM: {self.model_name}")
        logger.info(f"Device: {self.device}, Tensor Parallel Size: {self.tensor_parallel_size}")
        self.latency.start("model_loading")
        
        try:
            # 加载分词器，以支持对话模板
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # 如果尚未设置填充 token，则进行设置
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 初始化 vLLM 引擎
            self.llm = LLM(
                model=self.model_name,
                tensor_parallel_size=self.tensor_parallel_size,
                trust_remote_code=True,
                dtype="half" if self.device == "cuda" else "float32",
                max_model_len=16384,
            )
            
            elapsed = self.latency.stop("model_loading")
            logger.info(f"vLLM model loaded in {elapsed*1000:.2f} ms")
            
        except Exception as e:
            self.latency.stop("model_loading")
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _has_chat_template(self) -> bool:
        """检查分词器是否具有对话模板。"""
        if self.tokenizer is None:
            return False
        # 检查 chat_template 属性是否存在、是否不为 None 且非空
        return (hasattr(self.tokenizer, 'chat_template') and 
                self.tokenizer.chat_template is not None and
                len(self.tokenizer.chat_template) > 0)
    
    def _build_context_string(self, documents: List[Tuple[str, float]], 
                               context_prefix: str = "") -> str:
        """根据检索到的文档构造上下文字符串。"""
        context_parts = []
        for i, (doc, score) in enumerate(documents, 1):
            # 截断过长的文档
            doc_text = doc[:1000] if len(doc) > 1000 else doc
            context_parts.append(f"[Document {i}] {doc_text}")
        
        context = "\n\n".join(context_parts)
        return f"{context_prefix}{context}" if context_prefix else context
    
    def _build_rag_prompt(self, query: str, documents: List[Tuple[str, float]], 
                          context_prefix: str = "") -> str:
        """使用检索到的文档构造 RAG 提示词；如果有对话模板，则使用该模板。"""
        context = self._build_context_string(documents, context_prefix)
        
        # 构造用户消息内容
        user_content = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}"""

        # 检查模型是否具有对话模板
        if self._has_chat_template():
            logger.info("Using model's chat template for prompt formatting")
            messages = [
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context. Be concise and accurate."},
                {"role": "user", "content": user_content}
            ]
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                return prompt
            except Exception as e:
                logger.warning(f"Failed to apply chat template: {e}. Using default format.")
        
        # 回退到默认格式（不使用对话模板）
        logger.info("Using default prompt format (no chat template available)")
        prompt = f"""{user_content}

Answer:"""
        
        return prompt
    
    def _build_continuation_prompt(self, original_prompt: str, generated_text: str,
                                    new_docs: List[Tuple[str, float]]) -> str:
        """使用新检索到的文档，为逐句检索 RAG 构造续写提示词。"""
        # 根据新文档构造要插入的上下文
        context_parts = []
        for i, (doc, score) in enumerate(new_docs, 1):
            doc_text = doc[:500] if len(doc) > 500 else doc
            context_parts.append(f"[Doc {i}] {doc_text}")
        
        context_insert = "\n\n[Additional Context from Retrieval]\n" + "\n".join(context_parts) + "\n\n[Continue Answer]\n"
        
        # 对于使用对话模板的模型，需要采用不同的处理方式
        if self._has_chat_template():
            # 追加已生成文本和新上下文，然后继续生成
            return original_prompt + generated_text + context_insert
        else:
            return original_prompt + generated_text + context_insert
    
    def generate_simple(self, query: str, documents: List[Tuple[str, float]], 
                        max_new_tokens: int = 256) -> str:
        """使用 vLLM，通过简单 RAG 模式生成回答。"""
        if self.llm is None:
            raise RuntimeError("Generator not initialized. Call initialize() first.")
        
        prompt = self._build_rag_prompt(query, documents)
        
        logger.info("Generating response (simple RAG) with vLLM...")
        
        # 统计输入 token 数，用于日志记录
        input_tokens = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=16384)
        logger.info(f"device: {self.device}, input tokens: {input_tokens['input_ids'].shape[1]}")
        
        # 设置采样参数
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
        )
        
        self.latency.start("generation")
        
        # 使用 vLLM 生成文本
        outputs = self.llm.generate([prompt], sampling_params)
        
        elapsed = self.latency.stop("generation")
        
        # 提取生成的文本
        response = outputs[0].outputs[0].text
        num_tokens = len(outputs[0].outputs[0].token_ids)
    
        logger.info(f"Generation completed in {elapsed*1000:.2f} ms")
        logger.info(f"Generated {num_tokens} tokens")
        
        return response.strip()
    
    def generate_fix_sentence(self, query: str, retriever: BM25Retriever,
                              initial_docs: List[Tuple[str, float]],
                              max_new_tokens: int = 256,
                              retrieval_k: int = 3) -> str:
        """使用 vLLM，通过逐句检索 RAG 模式（每生成一句后再次检索）生成回答。"""
        if self.llm is None:
            raise RuntimeError("Generator not initialized. Call initialize() first.")
        
        logger.info("Generating response (fix-sentence RAG) with vLLM...")
        logger.info(f"Retrieval k for subsequent sentences: {retrieval_k}")
        
        # 使用检索到的文档构造初始提示词
        current_docs = initial_docs
        generated_text = ""
        current_sentence = ""
        sentence_count = 0
        total_tokens_generated = 0
        
        self.latency.start("generation_total")
        
        # 构造初始提示词
        prompt = self._build_rag_prompt(query, current_docs)
        current_prompt = prompt
        
        # 每次生成一个 token 的采样参数
        single_token_params = SamplingParams(
            max_tokens=1,
            temperature=0.7,
            top_p=0.9,
        )
        
        while total_tokens_generated < max_new_tokens:
            self.latency.start("token_generation")
            
            # 每次生成一个 token，以检测句子边界
            outputs = self.llm.generate([current_prompt + generated_text], single_token_params)
            
            self.latency.stop("token_generation")
            
            # 获取生成的 token
            if not outputs[0].outputs[0].token_ids:
                logger.info("No token generated, stopping")
                break
            
            new_token_id = outputs[0].outputs[0].token_ids[0]
            new_token = outputs[0].outputs[0].text
            
            # 检查是否遇到序列结束标记（EOS）
            if new_token_id == self.tokenizer.eos_token_id:
                logger.info("EOS token generated, stopping")
                break
            
            current_sentence += new_token
            generated_text += new_token
            total_tokens_generated += 1
            
            # 检查是否到达句末
            if any(char in new_token for char in self.sentence_end_chars):
                sentence_count += 1
                sentence_text = current_sentence.strip()
                
                if sentence_text:
                    logger.info(f"Sentence {sentence_count} completed: '{sentence_text[:100]}...'")
                    
                    # 使用已生成的句子检索新文档
                    logger.info(f"Retrieving documents for sentence {sentence_count}")
                    new_docs = retriever.retrieve(sentence_text, k=retrieval_k)
                    
                    if new_docs:
                        # 使用新上下文构造续写提示词
                        current_prompt = self._build_continuation_prompt(prompt, generated_text, new_docs)
                        # 清空 generated_text，因为它已包含在 current_prompt 中
                        generated_text = ""
                        
                        logger.info(f"Appended {len(new_docs)} new documents to context")
                    
                    current_sentence = ""
        
        elapsed = self.latency.stop("generation_total")
        logger.info(f"Fix-sentence generation completed in {elapsed*1000:.2f} ms")
        logger.info(f"Generated {total_tokens_generated} tokens, {sentence_count} sentences")
        
        return generated_text.strip()


class RAGPipeline:
    """组合检索器和生成器的 RAG 主流水线。"""
    
    def __init__(self, corpus_name: str, model_name: str, 
                 initial_k: int = 64, sentence_k: int = 3,
                 cache_dir: str = "./cache",
                 retriever_tokenizer: str = "EleutherAI/gpt-j-6b",
                 device: str = 'cuda'):
        self.retriever = BM25Retriever(corpus_name, cache_dir, retriever_tokenizer, device)
        self.generator = RAGGenerator(model_name)
        self.initial_k = initial_k
        self.sentence_k = sentence_k
        self.latency = LatencyTracker()
    
    def initialize(self):
        """初始化检索器和生成器。"""
        logger.info("=" * 60)
        logger.info("INITIALIZING RAG PIPELINE")
        logger.info("=" * 60)
        
        self.latency.start("pipeline_init")
        
        self.retriever.initialize()
        self.generator.initialize()
        
        elapsed = self.latency.stop("pipeline_init")
        logger.info(f"Pipeline initialization completed in {elapsed*1000:.2f} ms")
    
    def set_device(self, device: str):
        self.retriever.device = device
        self.retriever.initialize()
    
    def query(self, question: str, mode: str = "simple", 
              max_new_tokens: int = 256) -> str:
        """处理查询并生成回答。"""
        logger.info("=" * 60)
        logger.info(f"PROCESSING QUERY (mode: {mode})")
        logger.info(f"Question: {question}")
        logger.info("=" * 60)
        
        # 检索文档
        documents, kernel_time = self.retriever.retrieve(question, k=self.initial_k)
        
        self.latency.start("total_query")
        # 根据模式生成回答
        if mode == "simple":
            response = self.generator.generate_simple(question, documents, max_new_tokens)
        elif mode == "fix-sentence":
            response = self.generator.generate_fix_sentence(
                question, self.retriever, documents, 
                max_new_tokens, self.sentence_k
            )
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'simple' or 'fix-sentence'")
        
        elapsed = self.latency.stop("total_query")
        print(f"\nTotal query time: {elapsed*1000 + kernel_time:.2f} ms")
        
        return response
    
    def gen_warmup(self):
        logger.info("Starting generation warmup...")
        documents = self.retriever.get_documents([
            235906, 1329624, 3853890, 4033148, 1143300, 170388, 193570, 2129634, 1792798, 2421336, 1182126, 519672, 3936786, 409458, 4824994, 4138176, 2568248, 743084, 4896700, 3919388, 4168079, 1127326, 843730, 4681695, 3822656, 1176055, 2127298, 4108943, 2393310, 388004, 434690, 3988418, 1166386, 3151314, 4986968, 3019791, 363164, 1463090, 5012462, 993906, 2289340, 4755356, 1009676, 3827140, 921151, 2278300, 2909804, 1372754, 4568258, 3660158, 3430802, 3295214, 4920243, 3055680, 4726658, 446370, 1757627, 515163, 651794, 4716290, 950258
        ])
        for i in range(6):
            response = self.generator.generate_simple(
                """
                United States v. Cecil Price, et al., also known as the Mississippi Burning trial, was a criminal trial where the United States charged a group of 18 men with conspiring in a Ku Klux Klan plot to murder three young civil rights workers (Michael Schwerner, James Chaney, and Andrew Goodman) in Philadelphia, Mississippi on which date, the murders of Chaney, Goodman, and Schwerner, also known as the Freedom Summer murders, involved three activists that were abducted and murdered in Neshoba County, Mississippi, during the Civil Rights Movement? Answer in detail.
                """,
                documents,
                64
            )
            logger.info(response)

    
    def print_metrics(self):
        """打印所有延迟指标。"""
        logger.info("\n" + "=" * 60)
        logger.info("COMBINED METRICS")
        logger.info("=" * 60)
        
        self.latency.print_summary()
        
        logger.info("\nRETRIEVER METRICS:")
        self.retriever.latency.print_summary()
        
        logger.info("\nGENERATOR METRICS:")
        self.generator.latency.print_summary()


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="RAG Pipeline with BM25S and HuggingFace",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--corpus", "-c",
        type=str,
        default="BeIR/hotpotqa",
        help="HuggingFace dataset name for corpus"
    )

    parser.add_argument(
        "--device",
        type=str,
        default='cuda',
        help="Device to run the model on (e.g., 'cuda' or 'hybrid')"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="HuggingFace model name for generation"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["simple", "fix-sentence"],
        default="simple",
        help="RAG mode: 'simple' for single retrieval, 'fix-sentence' for per-sentence retrieval"
    )
    
    parser.add_argument(
        "--initial-k", "-k",
        type=int,
        default=64,
        help="Number of documents to retrieve for initial query"
    )
    
    parser.add_argument(
        "--sentence-k",
        type=int,
        default=3,
        help="Number of documents to retrieve for each sentence (fix-sentence mode)"
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum new tokens to generate"
    )
    
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./cache",
        help="Directory to cache tokenized corpus"
    )
    
    parser.add_argument(
        "--retriever-tokenizer",
        type=str,
        default="EleutherAI/gpt-j-6b",
        help="Tokenizer to use for BM25 retrieval (default: GPT-J-6B)"
    )
    
    parser.add_argument(
        "--question", "-q",
        type=str,
        default=None,
        help="Question to ask (if not provided, enters interactive mode)"
    )
    
    return parser.parse_args()

def fpga_retriever_setup(
    bitstream: str = "../indexer_bm25.xclbin",
    export_dir: str = "./export",
):
    
    # 加载文档频率（包含各词项的文档数量）
    logger.info("Loading doc_freq.bin...")
    doc_freq = load_document_frequency_mmap(os.path.join(export_dir, "doc_freq.bin"))
    
    if doc_freq is None:
        logger.error("Failed to load doc_freq.bin")
        return None
    
    # 加载词频
    logger.info("Loading term_freq.bin...")
    term_freq = load_term_frequencies_mmap(os.path.join(export_dir, "term_freq.bin"))

    if term_freq is None:
        logger.error("Failed to load term_freq.bin")
        return None
    
    # 将文档打包成硬件所需的格式
    packed = pack_documents_for_hw(term_freq, 8)
    
    # 计算 L 和 L_doc_total
    L = packed.num_docs
    L = ((L + 63) // 64) * 64  # 向上补齐到 64 的倍数
    L_doc_total = packed.vectors_per_channel()
    
    logger.info("\n======================================")
    logger.info("KERNEL LAUNCH PARAMETERS")
    logger.info("======================================")
    logger.info(f"L (num docs, padded): {L}")
    logger.info(f"L_doc_total (vectors per channel): {L_doc_total}")
    logger.info(f"Num super-batches: {packed.num_super_batches}")
    
    # ===============================
    # 初始化 PyXRT 设备和内核
    # ===============================
    
    logger.info("\n======================================")
    logger.info("PYXRT DEVICE AND KERNEL INITIALIZATION")
    logger.info("======================================")
    
    result = find_working_device(bitstream, -1)
    if result is None:
        logger.error(f" No working device found for XCLBIN: {bitstream}")
        logger.info("Please check:")
        logger.info("  1. FPGA devices are properly installed and visible")
        logger.info("  2. The XCLBIN file exists and is compatible with the device")
        logger.info("  3. XRT runtime is properly installed")
        return None
    
    device, xclbin_uuid, selected_device = result
    logger.info(f"Device {selected_device} opened and XCLBIN loaded")
    
    # 创建内核对象
    logger.info("Creating kernel object...")
    try:
        kernel = pyxrt.kernel(device, xclbin_uuid, "indexer_top")
    except Exception as e:
        logger.error(f" Failed to create kernel object: {e}")
        return None
    logger.info(f"Kernel created.")
    
    # ===============================
    # 分配缓冲区
    # ===============================
    
    logger.info("\n======================================")
    logger.info("BUFFER ALLOCATION")
    logger.info("======================================")
    
    # 计算缓冲区大小
    df_buffer_size = VOCAB_SIZE_DIV_16 * 16 * 4  # 每个向量包含 16 个 int
    query_bitmap_size = VOCAB_SIZE_DIV_512 * 64  # 每块 512 位，即 64 字节
    inst_mem_size = packed.num_super_batches * 4
    doc_mem_size = L_doc_total * 16 * 4  # 每个向量包含 16 个 uint32
    output_size = (TOP_K + 15) // 16
    topk_id_size = output_size * 16 * 4
    
    logger.info("Buffer sizes:")
    logger.info(f"  df_buffer: {df_buffer_size / 1024:.2f} KB")
    logger.info(f"  query_bitmap: {query_bitmap_size / 1024:.2f} KB")
    logger.info(f"  inst_mem: {inst_mem_size / 1024:.2f} KB")
    logger.info(f"  doc_mem (per channel): {doc_mem_size / 1024 / 1024:.2f} MB")
    logger.info(f"  topk_id: {topk_id_size} bytes")
    
    # 使用 kernel.group_id() 获取内存组分配信息并分配缓冲区
    # 参数顺序：L(0)、L_doc_total(1)、df_buffer(2)、query_bitmap(3)、inst_mem(4)、
    #                 doc_mem[0-3](5-8), topk_id(9)
    
    # 与 Xilinx 示例一样，使用零初始化
    zeros_df = bytearray(df_buffer_size)
    zeros_query = bytearray(query_bitmap_size)
    zeros_inst = bytearray(inst_mem_size)
    zeros_doc = bytearray(doc_mem_size)
    zeros_topk = bytearray(topk_id_size)
    
    logger.info("Allocate and initialize buffers")
    
    # 分配 df_buffer 缓冲区
    bo_df_buffer = pyxrt.bo(device, df_buffer_size, pyxrt.bo.normal, kernel.group_id(2))
    bo_df_buffer.write(zeros_df, 0)
    buf_df = bo_df_buffer.map()
    
    # 分配 query_bitmap 缓冲区
    bo_query_bitmap = pyxrt.bo(device, query_bitmap_size, pyxrt.bo.normal, kernel.group_id(3))
    bo_query_bitmap.write(zeros_query, 0)
    buf_query = bo_query_bitmap.map()
    
    # 分配 inst_mem 缓冲区
    bo_inst_mem = pyxrt.bo(device, inst_mem_size, pyxrt.bo.normal, kernel.group_id(4))
    bo_inst_mem.write(zeros_inst, 0)
    buf_inst = bo_inst_mem.map()
    
    # 分配各路 doc_mem 缓冲区
    bo_doc_mem_0 = pyxrt.bo(device, doc_mem_size, pyxrt.bo.normal, kernel.group_id(5))
    bo_doc_mem_0.write(zeros_doc, 0)
    buf_doc_0 = bo_doc_mem_0.map()
    
    bo_doc_mem_1 = pyxrt.bo(device, doc_mem_size, pyxrt.bo.normal, kernel.group_id(6))
    bo_doc_mem_1.write(zeros_doc, 0)
    buf_doc_1 = bo_doc_mem_1.map()
    
    bo_doc_mem_2 = pyxrt.bo(device, doc_mem_size, pyxrt.bo.normal, kernel.group_id(7))
    bo_doc_mem_2.write(zeros_doc, 0)
    buf_doc_2 = bo_doc_mem_2.map()
    
    bo_doc_mem_3 = pyxrt.bo(device, doc_mem_size, pyxrt.bo.normal, kernel.group_id(8))
    bo_doc_mem_3.write(zeros_doc, 0)
    buf_doc_3 = bo_doc_mem_3.map()
    
    # 分配 topk_id 输出缓冲区
    bo_topk_id = pyxrt.bo(device, topk_id_size, pyxrt.bo.normal, kernel.group_id(9))
    bo_topk_id.write(zeros_topk, 0)
    buf_topk = bo_topk_id.map()
    
    logger.info(f"Buffers allocated")
    
    # ===============================
    # 准备数据并写入缓冲区
    # ===============================
    
    logger.info("\n======================================")
    logger.info("DATA PREPARATION AND TRANSFER")
    logger.info("======================================")
    
    # 准备并写入 df_buffer 数据
    logger.info("Writing df_buffer data...")
    df_buffer_data = np.zeros(VOCAB_SIZE_DIV_16 * 16, dtype=np.int32)
    for i in range(VOCAB_SIZE_DIV_16):
        for j in range(16):
            df_buffer_data[i * 16 + j] = int(doc_freq[i * 16 + j])
    # 使用 bo.write() 方法写入
    bo_df_buffer.write(df_buffer_data.tobytes(), 0)
    
    # 准备并写入 inst_mem
    logger.info("Writing inst_mem data...")
    inst_mem_data = np.array(packed.inst_mem, dtype=np.int32)
    bo_inst_mem.write(inst_mem_data.tobytes(), 0)
    
    # 准备并写入每个通道的 doc_mem
    logger.info("Writing doc_mem data for 4 channels...")
    doc_mem_buffers = [buf_doc_0, buf_doc_1, buf_doc_2, buf_doc_3]
    doc_mem_bos = [bo_doc_mem_0, bo_doc_mem_1, bo_doc_mem_2, bo_doc_mem_3]
    
    for channel in range(4):
        channel_data = np.zeros(L_doc_total * 16, dtype=np.uint32)
        for vec_idx, vec in enumerate(packed.doc_mem[channel]):
            for j in range(16):
                channel_data[vec_idx * 16 + j] = vec[j]
        doc_mem_bos[channel].write(channel_data.tobytes(), 0)
    
    # 将输出缓冲区初始化为 -1
    logger.info("Initializing topk_id output buffer...")
    topk_id_data = np.full(output_size * 16, -1, dtype=np.int32)
    bo_topk_id.write(topk_id_data.tobytes(), 0)
    
    logger.info(f"  output_size: {output_size}, topk_id_size: {topk_id_size} bytes")
    
    # 将缓冲区同步到设备
    logger.info("Syncing buffers to device...")
    
    bo_df_buffer.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, df_buffer_size, 0)
    bo_inst_mem.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, inst_mem_size, 0)
    bo_doc_mem_0.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, doc_mem_size, 0)
    bo_doc_mem_1.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, doc_mem_size, 0)
    bo_doc_mem_2.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, doc_mem_size, 0)
    bo_doc_mem_3.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, doc_mem_size, 0)

    return kernel, L, L_doc_total, bo_query_bitmap, bo_df_buffer, bo_inst_mem, bo_doc_mem_0, bo_doc_mem_1, bo_doc_mem_2, bo_doc_mem_3, bo_topk_id, buf_topk

def fpga_retriver_launch(
    kernel, L, L_doc_total, bo_query_bitmap, bo_df_buffer, bo_inst_mem, bo_doc_mem_0, bo_doc_mem_1, bo_doc_mem_2, bo_doc_mem_3, bo_topk_id, buf_topk, query_token_list
):
    query_tokens = parse_query_tokens(query_token_list)
    query_bitmap_size = VOCAB_SIZE_DIV_512 * 64  # 每块 512 位，即 64 字节
    output_size = (64 + 15) // 16
    topk_id_size = output_size * 16 * 4
    logger.info(f"\nParsed {len(query_tokens)} query tokens from input")
    # 准备查询位图
    logger.info("Preparing query bitmap...")
    query_bitmap = np.zeros(VOCAB_SIZE_DIV_512 * 64, dtype=np.uint8)
    for token_id in query_tokens:
        tid = int(token_id)
        if 0 <= tid < VOCAB_SIZE:
            chunk_idx = tid // 512
            bit_idx = tid % 512
            byte_idx = bit_idx // 8
            bit_in_byte = bit_idx % 8
            query_bitmap[chunk_idx * 64 + byte_idx] |= (1 << bit_in_byte)
    
    # 将查询位图写入缓冲区
    bo_query_bitmap.write(query_bitmap.tobytes(), 0)
    
    # 将查询位图同步到设备
    bo_query_bitmap.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, query_bitmap_size, 0)
    
    # 启动内核
    logger.info("Launching kernel...")
    start_time = time.time()
    
    run = kernel(
        L,
        L_doc_total,
        bo_df_buffer,
        bo_query_bitmap,
        bo_inst_mem,
        bo_doc_mem_0,
        bo_doc_mem_1,
        bo_doc_mem_2,
        bo_doc_mem_3,
        bo_topk_id
    )
    
    logger.info("Now wait for the kernel to finish")
    state = run.wait()
    
    kernel_time = (time.time() - start_time) * 1000  # 单位为毫秒
    logger.info(f"Kernel execution completed in {kernel_time:.2f} ms")
    logger.info(f"  Kernel state: {state}")

    if state != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
        logger.warning(f" Kernel did not complete successfully! State: {state}")
    
    
    # 将输出缓冲区从设备同步回主机
    bo_topk_id.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, topk_id_size, 0)
    
    # 读取 Top-K 文档 ID
    hw_topk_indices = []
    topk_bytes = bytes(buf_topk[:topk_id_size])
    topk_result = np.frombuffer(topk_bytes, dtype=np.int32)

    for i in range(output_size):
        for j in range(16):
            if i * 16 + j < 64:
                hw_topk_indices.append(int(topk_result[i * 16 + j]))
    
    return hw_topk_indices, kernel_time
    
def main():
    """程序主入口。"""
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("RAG PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Corpus: {args.corpus}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Retriever Tokenizer: {args.retriever_tokenizer}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Initial K: {args.initial_k}")
    logger.info(f"Sentence K: {args.sentence_k}")
    logger.info(f"Max tokens: {args.max_tokens}")
    logger.info(f"Cache dir: {args.cache_dir}")
    logger.info("=" * 60)
    
    # 初始化流水线
    pipeline = RAGPipeline(
        corpus_name=args.corpus,
        model_name=args.model,
        initial_k=args.initial_k,
        sentence_k=args.sentence_k,
        cache_dir=args.cache_dir,
        retriever_tokenizer=args.retriever_tokenizer,
        device=args.device
    )
    
    pipeline.initialize()

    #pipeline.gen_warmup()
    
    if args.question:
        # 单次问答模式
        response = pipeline.query(args.question, mode=args.mode, 
                                  max_new_tokens=args.max_tokens)
        
        logger.info("=" * 60)
        logger.info("RESPONSE")
        logger.info("=" * 60)
        print(f"\n{response}\n")
        
        pipeline.print_metrics()
    else:
        # 交互模式
        print("\nEntering interactive mode. Type 'quit' or 'exit' to stop.")
        print("Type 'metrics' to show latency metrics.")
        print("Type 'warmup' to run generation warmup.")
        print("Type 'mode simple' or 'mode fix-sentence' to change mode.")
        print("-" * 60)
        
        current_mode = args.mode
        
        while True:
            try:
                print(f"\n[{current_mode}] Enter your question: ", end="")
                user_input = input().strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ["quit", "exit"]:
                    logger.info("Exiting...")
                    break
                
                if user_input.lower() == "metrics":
                    pipeline.print_metrics()
                    continue
                
                if user_input.lower() == "warmup":
                    pipeline.gen_warmup()
                    continue
                
                if user_input.lower().startswith("mode "):
                    new_mode = user_input[5:].strip()
                    if new_mode in ["simple", "fix-sentence"]:
                        current_mode = new_mode
                        logger.info(f"Mode changed to: {current_mode}")
                    else:
                        logger.warning("Invalid mode. Use 'simple' or 'fix-sentence'")
                    continue
                
                if user_input.lower().startswith("device "):
                    new_device = user_input[7:].strip()
                    pipeline.set_device(new_device)
                    print(f"Device changed to: {new_device}")
                    continue

                # 处理用户问题
                response = pipeline.query(user_input, mode=current_mode,
                                          max_new_tokens=args.max_tokens)
                
                print("\n" + "=" * 60)
                print("RESPONSE")
                print("=" * 60)
                print(f"\n{response}\n")
                
            except KeyboardInterrupt:
                print()
                logger.info("Interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                continue
        
        # 打印最终统计指标
        pipeline.print_metrics()


if __name__ == "__main__":
    main()
