# 段 4b: 崩れた図表ページを画像にして、画像対応モデルで Markdown 表に起こす
# 使い方  python src/figure.py render|read [モデル]|build
import base64
import json
import sys
import time

import fitz
import requests

MODEL = "qwen2.5vl:7b"
PROMPT = """この画像は日本語の白書の1ページです。ページ内の図・グラフ・表を全て、次の形で Markdown に起こしてください。
### 図表の題(図表番号つき)
| 項目 | 系列(凡例) | 数値(単位) |
- 棒・線・表の数値はページに印字されている数字だけを写す。読めない・印字が無い数値は「不明」と書き、推測しない。
- 本文の文章は写さない。図表が無ければ「図表なし」とだけ書く。"""


def render():
    # 崩れたページを 170dpi の画像にする
    doc = fitz.open("data/raw/all.pdf")
    for p in json.load(open("data/bad_pages.json")):
        doc[p - 1].get_pixmap(dpi=170).save(f"data/figimg/p{p}.png")


def read(model):
    # 1 ページずつ VLM に読ませ、結果を data/figures.jsonl に残す(途中再開できる)
    out = "data/figures.jsonl"
    done = set()
    try:
        done = {json.loads(l)["pdf_page"] for l in open(out, encoding="utf-8")}
    except FileNotFoundError:
        pass
    for p in json.load(open("data/bad_pages.json")):
        if p in done:
            continue
        img = base64.b64encode(open(f"data/figimg/p{p}.png", "rb").read()).decode()
        t = time.time()
        r = requests.post("http://127.0.0.1:11434/api/chat", json={
            "model": model, "stream": False, "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 2000},
            "messages": [{"role": "user", "content": PROMPT, "images": [img]}]}, timeout=900)
        r.raise_for_status()
        text = r.json()["message"]["content"].strip()
        print(p, f"{time.time() - t:.0f}s", len(text), flush=True)
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps({"pdf_page": p, "model": model, "text": text}, ensure_ascii=False) + "\n")


def build():
    # 基準の chunks/embeddings を写して、図表の断片(kind=figure)と埋め込みを足す(_f 版)
    import numpy as np
    from search import embed
    chunks = [json.loads(l) for l in open("data/chunks.jsonl", encoding="utf-8")]
    figs = [json.loads(l) for l in open("data/figures.jsonl", encoding="utf-8")]
    new = []
    for f in figs:
        if len(f["text"]) < 60:  # 図表なし(短い返答)は足さない
            continue
        new.append({"id": len(chunks) + len(new), "pdf_page": f["pdf_page"], "paper_page": f["pdf_page"] - 13,
                    "kind": "figure", "text": "【図表を画像から起こした内容】\n" + f["text"]})
    vecs = np.vstack([np.load("data/embeddings.npy"), np.vstack([embed([c["text"] for c in new[i:i + 8]]) for i in range(0, len(new), 8)])])
    with open("data/chunks_f.jsonl", "w", encoding="utf-8") as fo:
        for c in chunks + new:
            fo.write(json.dumps(c, ensure_ascii=False) + "\n")
    np.save("data/embeddings_f.npy", vecs)
    print("足した断片", len(new), "合計", len(chunks) + len(new), vecs.shape)


if __name__ == "__main__":
    {"render": render, "read": lambda: read(sys.argv[2] if len(sys.argv) > 2 else MODEL), "build": build}[sys.argv[1]]()
