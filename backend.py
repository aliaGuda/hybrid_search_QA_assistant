import json
import re
import os
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI  # OpenRouter uses the OpenAI-compatible API format


class HybridSearchRAG:
    """
    Encapsulates Hybrid Search (BM25 + Semantic) and RAG via OpenRouter.
    """

    def __init__(self, data_paths=None):
        """
        Args:
            data_paths (list): Paths to the JSON dataset files.
        """
        self.documents = []          # Parsed document dicts
        self.doc_texts = []          # Plain text list for indexing
        self.bm25 = None             # BM25 model
        self.st_model = None         # SentenceTransformer model
        self.doc_embeddings = None   # Dense document embeddings
        self._or_client = None       # OpenRouter API client
        self._or_model = None        # OpenRouter model name

        if data_paths:
            self.load_data(data_paths)
            self.build_bm25()
            self.build_semantic_index()

    # ------------------------------------------------------------------
    # Data Loading
    # ------------------------------------------------------------------
    def clean_html(self, text):
        """Strips HTML tags from raw text."""
        return re.sub(re.compile('<.*?>'), '', text).strip()

    def load_data(self, data_paths):
        """Loads and parses the JSON dataset files."""
        self.documents = []
        for path in data_paths:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    for item in json.load(f):
                        title = item.get('title', '')
                        body  = self.clean_html(item.get('body', ''))
                        self.documents.append({
                            'id':    item.get('question_id', ''),
                            'title': title,
                            'body':  body,
                            'text':  f"{title}. {body}",
                            'link':  item.get('link', '')
                        })
        self.doc_texts = [d['text'] for d in self.documents]
        print(f"Loaded {len(self.documents)} documents.")

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    def build_bm25(self):
        """Builds the BM25 lexical index."""
        print("Building BM25 index...")
        self.bm25 = BM25Okapi([d.lower().split() for d in self.doc_texts])

    def build_semantic_index(self, model_name='all-MiniLM-L6-v2'):
        """Encodes all documents into dense embeddings."""
        print(f"Loading semantic model: {model_name}...")
        self.st_model = SentenceTransformer(model_name)
        print("Computing embeddings (takes ~1-2 min on Colab T4)...")
        self.doc_embeddings = self.st_model.encode(
            self.doc_texts, convert_to_tensor=True, show_progress_bar=True
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def search_bm25(self, query, top_k=10):
        """BM25 lexical search. Returns ranked list of dicts."""
        scores   = self.bm25.get_scores(query.lower().split())
        top_idx  = np.argsort(scores)[::-1][:top_k]
        return [{'doc_id': self.documents[i]['id'],
                 'doc':    self.documents[i],
                 'score':  scores[i],
                 'rank':   r + 1}
                for r, i in enumerate(top_idx)]

    def search_semantic(self, query, top_k=10):
        """Dense semantic search using cosine similarity."""
        q_emb      = self.st_model.encode(query, convert_to_tensor=True)
        cos_scores = util.cos_sim(q_emb, self.doc_embeddings)[0]
        top        = torch.topk(cos_scores, k=top_k)
        return [{'doc_id': self.documents[idx]['id'],
                 'doc':    self.documents[idx],
                 'score':  score.item(),
                 'rank':   r + 1}
                for r, (score, idx) in enumerate(zip(top[0], top[1]))]

    def _rrf(self, rank, k=60):
        """Reciprocal Rank Fusion score."""
        return 1 / (k + rank)

    def search_hybrid(self, query, top_k=10):
        """Fuses BM25 and Semantic results with RRF."""
        scores, docs = {}, {}
        for res in self.search_bm25(query, 50) + self.search_semantic(query, 50):
            did = res['doc_id']
            docs[did]   = res['doc']
            scores[did] = scores.get(did, 0) + self._rrf(res['rank'])
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [{'doc_id': did, 'doc': docs[did], 'score': s, 'rank': r + 1}
                for r, (did, s) in enumerate(ranked)]

    # ------------------------------------------------------------------
    # RAG via OpenRouter
    # ------------------------------------------------------------------
    def load_rag_model(self, api_key, model_name="openai/gpt-oss-120b:free"):
        """
        Initialises the OpenRouter client.
        Get a free key at https://openrouter.ai (no credit card).
        """
        self._or_model  = model_name
        self._or_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        print(f"OpenRouter client ready. Requested model: {model_name}")

    def generate_answer(self, query, retrieved_docs):
        """
        Generates a cited answer using OpenRouter.

        Returns:
            str: The LLM-generated answer.
        """
        if not self._or_client:
            raise RuntimeError("Call load_rag_model(api_key) first.")

        # Build numbered source blocks for clear attribution
        source_blocks = []
        for i, res in enumerate(retrieved_docs[:3]):
            snippet = res['doc']['text'][:500].replace('\n', ' ')
            source_blocks.append(
                f"[Source {i+1}] Title: {res['doc']['title']}\nContent: {snippet}"
            )

        context = "\n\n".join(source_blocks)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful technical assistant specializing in AI and Data Science. "
                    "Answer the user's question using ONLY the provided sources. "
                    "Be clear and complete. Cite sources at the end like [Source 1]."
                )
            },
            {
                "role": "user",
                "content": f"Sources:\n\n{context}\n\nQuestion: {query}\n\nAnswer:"
            }
        ]

        response = self._or_client.chat.completions.create(
            model=self._or_model,
            messages=messages,
            max_tokens=400,
            temperature=0.3,
        )

        # Print the actual model OpenRouter selected (important when using 'auto')
        print(f"[OpenRouter] Model actually used: {response.model}")

        return response.choices[0].message.content.strip()
