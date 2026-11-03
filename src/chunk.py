# 断片づくり: pages.jsonl を読み、長いページは 600 字ずつ（重なり 100 字）に切って chunks.jsonl に書く
import json
import re
import sys
import unicodedata

MAX_LEN = 600       # 1 断片の最大文字数
OVERLAP = 100       # 隣の断片と重ねる文字数
PAGE_GAP = 13       # 紙面ページ = PDF ページ - 13（段 1 で 33→20, 69→56, 292→279 と確認）
SKIP_PAGES = {6, 7, 8, 9}  # 目次ページは検索対象から外す（段 1 の記録）


def join_lines(text):
    # 文の途中の改行を消す（「。」で終わる行の改行だけ残す）
    text = re.sub(r"(?<!。)\n", "", text)
    return text.strip()


# 見出しの行: 「1．題」「（1）題」「第3節 題」。短い行だけ(本文の行を除くため)
SUB_RE = re.compile(r"^(\d{1,2}[.．]\s*\D.{1,35}|[（(]\d{1,2}[）)]\S.{1,35})$")
SEC_RE = re.compile(r"^第\s*\d+\s*節\s*\S.{1,30}$")


def find_headings(raw):
    # 生のページ本文から、節の題(あれば)と、小見出しの一覧(行そのもの)を取り出す
    section, subs = None, []
    for line in raw.splitlines():
        norm = unicodedata.normalize("NFKC", line).strip()
        if len(norm) > 40 or norm.endswith("。"):
            continue
        if SEC_RE.match(norm):
            section = norm
        elif SUB_RE.match(norm):
            subs.append(line.strip())
    return section, subs


def split_text(text):
    # 文字数上限を超えたら、重なりを持たせて切る。(開始位置, 断片)の組を返す
    if len(text) <= MAX_LEN:
        return [(0, text)]
    parts = []
    start = 0
    while start < len(text):
        parts.append((start, text[start:start + MAX_LEN]))
        if start + MAX_LEN >= len(text):
            break
        start += MAX_LEN - OVERLAP
    return parts


def main(heading=False):
    chunks = []
    section, last_sub = "", ""  # 直前までに見た節の題と小見出し(ページをまたいで引き継ぐ)
    with open("data/pages.jsonl", encoding="utf-8") as f:
        for line in f:
            page = json.loads(line)
            if page["page"] in SKIP_PAGES:
                continue
            text = join_lines(page["text"])
            if len(text) < 20:  # 表紙などほぼ空のページは捨てる
                continue
            sec, subs = find_headings(page["text"])
            if sec:
                section = sec
            pos = [(text.find(h), h) for h in subs if text.find(h) >= 0]
            for start, part in split_text(text):
                if heading:
                    # 断片の開始より前にある最後の小見出しを、頭に付ける
                    before = [h for p, h in pos if p <= start]
                    sub = before[-1] if before else last_sub
                    part = f"【{section}／{sub}】" + part
                chunks.append({
                    "id": len(chunks),
                    "pdf_page": page["page"],
                    "paper_page": page["page"] - PAGE_GAP,
                    "text": part,
                })
            if pos:
                last_sub = pos[-1][1]  # このページの最後の小見出しを次のページへ
    out = "data/chunks_h.jsonl" if heading else "data/chunks.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"断片数: {len(chunks)}")


if __name__ == "__main__":
    main(heading="--heading" in sys.argv)
