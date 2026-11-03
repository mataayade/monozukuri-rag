# 評価: 使い方
#   python src/evaluate.py run <AI名> <名前> [--heading] [--calc] [--k N]   40 問を流して results/<名前>_<AI名>.jsonl に記録
#   python src/evaluate.py report <名前> [<名前> ...]                       判定(src/judge.py の出力)と記録から集計表を出す
import json
import sys

from answer import answer, MODELS
from search import Searcher

LABELS = ("正解", "部分正解", "不正解", "でっち上げ")
GROUP_NAMES = {"para": "言い換え", "multi": "複数ページ", "table": "表・図のみ", "calc": "計算", "absent": "近い話題なし"}


def load_eval():
    with open("data/eval.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def group(r):
    # 旧 20 問は number/text/absent、新 20 問(h)は 5 つの種類で分ける
    if r["id"].startswith("h"):
        return GROUP_NAMES[r["type"]]
    return {"n": "旧 number", "t": "旧 text", "a": "旧 absent"}[r["id"][0]]


def hits(q, chunks):
    # ページ単位: 正解ページの断片が入ったか / 断片単位: 正解の語句(keys)が全部、入った断片のどれかに含まれるか
    if q["type"] == "absent":
        return None, None
    page = any(c["pdf_page"] in q["pages"] for c in chunks)
    texts = ["".join(c["text"].split()) for c in chunks]
    chunk = all(any("".join(k.split()) in t for t in texts) for k in q["keys"])
    return page, chunk


def run(ai, tag, heading, calc, k, figure=False):
    # 40 問を流して、検索結果・回答・時間・費用を記録する
    searcher = Searcher("data/chunks_h.jsonl", "data/embeddings_h.npy") if heading else Searcher()
    if figure:  # 段 4b: 図表を画像から起こした断片を足した版
        searcher = Searcher("data/chunks_f.jsonl", "data/embeddings_f.npy")
    out = []
    for q in load_eval():
        chunks = searcher.search(q["question"], k=k)
        page, chunk = hits(q, chunks)
        text, sec, cost, errors, model, used = answer(ai, q["question"], chunks, calc)
        print(q["id"], page, chunk, f"{sec:.1f}s ${cost:.5f}", errors or "", flush=True)
        out.append({"id": q["id"], "type": q["type"], "question": q["question"], "gold": q["answer"],
                    "gold_pages": q["pages"], "model": model, "hit_page": page, "hit_chunk": chunk,
                    "settings": {"heading": heading, "calc": calc, "k": k},
                    "retrieved_chunk_ids": [c["id"] for c in chunks],
                    "retrieved_pdf_pages": [c["pdf_page"] for c in chunks],
                    "answer": text, "used_idx": used, "cited_pdf_pages": [chunks[i]["pdf_page"] for i in used], "calc_errors": errors, "sec": sec, "cost": cost})
    with open(f"results/{tag}_{ai}.jsonl", "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load(tag, ai):
    # 記録に判定(src/judge.py の出力)を結びつけて返す
    rows = [json.loads(l) for l in open(f"results/{tag}_{ai}.jsonl", encoding="utf-8")]
    judge = {}
    for l in open(f"results/判定_{tag}_{ai}.jsonl", encoding="utf-8"):
        j = json.loads(l)
        judge[j["id"]] = j
    for r in rows:
        r["label"] = judge[r["id"]]["label"]
        r["reason"] = judge[r["id"]]["reason"]
    return rows


def summary(rows):
    # 1 行ぶんの集計: Hit@5(ページ/断片)、正解、部分正解、でっち上げ、秒/問
    n = len(rows)
    hp = [r["hit_page"] for r in rows if r["hit_page"] is not None]
    hc = [r["hit_chunk"] for r in rows if r["hit_chunk"] is not None]
    cnt = {x: sum(r["label"] == x for r in rows) for x in LABELS}
    sec = sum(r["sec"] for r in rows) / n
    return (f"{sum(hp)}/{len(hp)} | {sum(hc)}/{len(hc)} | {cnt['正解']}/{n} ({cnt['正解'] / n:.0%}) | "
            f"{cnt['部分正解']} | {cnt['不正解']} | {cnt['でっち上げ']} | {sec:.1f} | ${sum(r['cost'] for r in rows):.3f}")


def report(tags):
    # 名前ごと・AI ごとの全体表と、種類別の正解数を出す
    head = "| 条件 | AI | Hit@5 ページ | Hit@5 断片 | 正解 | 部分 | 不正解 | でっち上げ | 秒/問 | 費用 |\n|---|---|---|---|---|---|---|---|---|---|"
    print(head)
    data = {}
    for tag in tags:
        for ai in ("qwen3-8b", "qwen3-14b", "grok", "deepseek"):
            try:
                rows = load(tag, ai)
            except FileNotFoundError:
                continue
            data[(tag, ai)] = rows
            print(f"| {tag} | {ai} | {summary(rows)} |")
    groups = ["旧 number", "旧 text", "旧 absent", "言い換え", "複数ページ", "表・図のみ", "計算", "近い話題なし"]
    print("\n種類別の正解数(正解 / 問数)\n")
    print("| 条件 | AI | " + " | ".join(groups) + " |\n|---|---|" + "---|" * len(groups))
    for (tag, ai), rows in data.items():
        cells = []
        for g in groups:
            s = [r for r in rows if group(r) == g]
            cells.append(f"{sum(r['label'] == '正解' for r in s)}/{len(s)}")
        print(f"| {tag} | {ai} | " + " | ".join(cells) + " |")
    print("\n全体を旧 20 問 / 新 20 問に分けた正解数\n")
    for (tag, ai), rows in data.items():
        old = [r for r in rows if not r["id"].startswith("h")]
        new = [r for r in rows if r["id"].startswith("h")]
        f = lambda s: f"{sum(r['label'] == '正解' for r in s)}/{len(s)}"
        print(f"- {tag} {ai}: 旧 {f(old)} / 新 {f(new)}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "run":
        k = int(a[a.index("--k") + 1]) if "--k" in a else 5
        run(a[1], a[2], "--heading" in a, "--calc" in a, k, "--figure" in a)
    else:
        report(a[1:])
