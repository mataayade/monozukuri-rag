# 検索: 文字 2-gram の BM25 と bge-m3 の埋め込み(コサイン類似度)を、順位の統合(RRF)で合わせる
import json
import math
import os
from collections import Counter

import numpy as np
import requests

OLLAMA = "http://127.0.0.1:11434"
EMB_PATH = "data/embeddings.npy"


def load_chunks(path="data/chunks.jsonl"):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def bigrams(text):
    # 空白を除いて、2 文字ずつずらして切る（辞書不要の日本語の語切り）
    text = "".join(text.split())
    return [text[i:i + 2] for i in range(len(text) - 1)]


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(bigrams(d)) for d in docs]      # 断片ごとの語の出現数
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avg = sum(self.lens) / len(self.lens)
        df = Counter()
        for tf in self.tfs:
            df.update(tf.keys())                              # 語が何断片に出るか
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query):
        out = np.zeros(len(self.tfs))
        for t in set(bigrams(query)):
            if t not in self.idf:
                continue
            for i, tf in enumerate(self.tfs):
                f = tf.get(t, 0)
                if f:
                    out[i] += self.idf[t] * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg))
        return out


def embed(texts):
    # Ollama の bge-m3 で文章を数値の列に直す
    r = requests.post(f"{OLLAMA}/api/embed", json={"model": "bge-m3", "input": texts}, timeout=600)
    r.raise_for_status()
    return np.array(r.json()["embeddings"], dtype=np.float32)


def load_embeddings(chunks, emb_path=EMB_PATH):
    # 保存済みならそれを使い、無ければ計算して保存する（再計算しない）
    if os.path.exists(emb_path):
        return np.load(emb_path)
    vecs = []
    for i in range(0, len(chunks), 16):
        vecs.append(embed([c["text"] for c in chunks[i:i + 16]]))
        print(f"埋め込み {i + 16}/{len(chunks)}", flush=True)
    vecs = np.vstack(vecs)
    np.save(emb_path, vecs)
    return vecs


class Searcher:
    def __init__(self, chunks_path="data/chunks.jsonl", emb_path=EMB_PATH):
        self.chunks = load_chunks(chunks_path)
        self.bm25 = BM25([c["text"] for c in self.chunks])
        vecs = load_embeddings(self.chunks, emb_path)
        self.vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)  # 長さ 1 にそろえる

    def search(self, query, k=5, rrf_k=60):
        bm = self.bm25.scores(query)
        q = embed([query])[0]
        q = q / np.linalg.norm(q)
        cos = self.vecs @ q                                   # コサイン類似度
        # RRF: 各方式の順位 r から 1/(rrf_k + r) を足して統合
        fused = np.zeros(len(self.chunks))
        for s in (bm, cos):
            for rank, idx in enumerate(np.argsort(-s)):
                fused[idx] += 1.0 / (rrf_k + rank + 1)
        top = np.argsort(-fused)[:k]
        return [self.chunks[i] for i in top]


if __name__ == "__main__":
    s = Searcher()
    for c in s.search("製造業の労働時間は減りましたか"):
        print(c["paper_page"], c["text"][:60])
