# PDF から 1 ページずつ文字と表を取り出して jsonl に保存する
import json
import pymupdf  # PyMuPDF

PDF = "data/raw/all.pdf"

doc = pymupdf.open(PDF)  # PDF を開く
n_tables = 0
with open("data/pages.jsonl", "w", encoding="utf-8") as fp, \
     open("data/tables.jsonl", "w", encoding="utf-8") as ft:
    for i, page in enumerate(doc, start=1):  # PDF 上のページ番号は 1 から
        text = page.get_text()  # ページの文字を取り出す
        fp.write(json.dumps({"file": "all.pdf", "page": i, "text": text}, ensure_ascii=False) + "\n")
        for t in page.find_tables().tables:  # ページ内の表を探す
            md = t.to_markdown()  # 表を markdown にする
            ft.write(json.dumps({"file": "all.pdf", "page": i, "markdown": md}, ensure_ascii=False) + "\n")
            n_tables += 1
print("pages:", len(doc), "tables:", n_tables)
