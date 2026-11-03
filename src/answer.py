# 回答: 検索した上位断片をプロンプトに入れ、選んだ AI に根拠ページつきで答えさせる
import ast
import operator
import os
import re
import subprocess
import time
import unicodedata

import requests

# 回答 AI の名前 → 実際のモデル ID（安価なもの）。結果の記録に使う
MODELS = {
    "qwen3-8b": "qwen3:8b",
    "qwen3-14b": "qwen3:14b",  # 追加の比較用(ローカル)
    "grok": "grok",          # ask.py 経由(実際のモデル名は記録の model に残す)
    "deepseek": "deepseek",  # ask.py 経由
    "gemini": "gemini-2.5-flash-lite",
    "claude": "claude-haiku-4-5",
}

PROMPT = """あなたは「ものづくり基盤技術の振興施策（ものづくり白書）」についての質問に答えるアシスタントです。
次のルールを守ってください。
- 下の【資料】に書かれていることだけで答える。資料に無いことは推測しない。
- 資料に答えが書かれていない場合は「資料には書かれていません」とだけ答える。
- 数字の計算が必要な場合は、式を書いて計算する。
- 答えの最後の行に、使った断片の番号だけを [1][3] の形で書く。ページ番号は書かない。

【資料】
{context}

【質問】
{question}
"""


# 改善 b: 計算は AI にさせず、式だけ書かせて Python で計算する
CALC_RULE = "- 数字の計算が必要な場合は、自分で計算せず、式だけを [[計算: 318300-306000]] の形で書く(数字は半角、使える記号は + - * / と括弧だけ)。計算結果は書かない。式は後でプログラムが計算して置き換える。"
OLD_RULE = "- 数字の計算が必要な場合は、式を書いて計算する。"
PROMPT_CALC = PROMPT.replace(OLD_RULE, CALC_RULE)
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def safe_eval(expr):
    # 四則演算と括弧だけを許す自前の計算。それ以外の式は ValueError(eval は使わない)
    def walk(n):
        if isinstance(n, ast.Expression):
            return walk(n.body)
        if isinstance(n, ast.Constant) and type(n.value) in (int, float):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in OPS:
            return OPS[type(n.op)](walk(n.left), walk(n.right))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return -walk(n.operand)
        raise ValueError(f"許可しない式: {expr}")
    return walk(ast.parse(expr, mode="eval"))


def apply_calc(text):
    # 回答中の [[計算: 式]] を Python の計算結果に置き換える。評価できない式は件数に数える
    errors = []

    def repl(m):
        expr = unicodedata.normalize("NFKC", m.group(1)).replace(",", "").replace("×", "*").replace("÷", "/").replace("−", "-")
        try:
            v = safe_eval(expr)
        except (ValueError, SyntaxError, ZeroDivisionError) as e:
            errors.append(f"{expr}: {e}")
            return f"(計算式を評価できません: {expr})"
        v = round(v, 4)
        return f"{v:,.4f}".rstrip("0").rstrip(".")
    return re.sub(r"\[\[計算[:：]\s*(.+?)\]\]", repl, text), errors


def build_prompt(question, chunks, calc=False):
    # 断片に紙面ページ番号を付けて並べる
    context = "\n\n".join(f"[p.{c['paper_page']}]\n{c['text']}" for c in chunks)
    return (PROMPT_CALC if calc else PROMPT).format(context=context, question=question)


def attach_evidence(text, chunks):
    # 回答中の [n] を取り除き、使われた断片の紙面ページを「根拠: p.118」で末尾に付ける。使われた断片の番号(0 始まり)も返す
    used = []
    for n in re.findall(r"\[(\d+)\]", text):
        if 1 <= int(n) <= len(chunks) and int(n) - 1 not in used:
            used.append(int(n) - 1)
    body = re.sub(r"\s*\[\d+\]", "", text)
    body = re.sub(r"\s*根拠[:：].*$", "", body).strip()
    if re.search(r"資料(に|には)書かれていません", body):
        return body, []
    pages = list(dict.fromkeys(chunks[i]["paper_page"] for i in used))
    return body + "\n\n根拠: " + (", ".join(f"p.{p}" for p in pages) if pages else "根拠ページ不明"), used


def ask_ollama(prompt, ai="qwen3-8b"):
    # ローカルの qwen3:8b。think=False で思考出力をオフにする
    r = requests.post("http://127.0.0.1:11434/api/chat", json={
        "model": MODELS[ai], "think": False, "stream": False,
        "options": {"temperature": 0, "num_ctx": 8192},
        "messages": [{"role": "user", "content": prompt}]}, timeout=600)
    r.raise_for_status()
    d = r.json()
    return d["message"]["content"], d.get("prompt_eval_count", 0), d.get("eval_count", 0)


ASK_PY = os.path.expanduser("~/.claude/scripts/ask.py")


def ask_cli(prompt, ai):
    # ask.py を呼ぶ。標準出力が回答、標準エラーの最後の行に model と cost が出る
    for attempt in range(3):  # 通信の一時失敗は最大 3 回まで試し、毎回表示する(3 回失敗したら止める)
        p = subprocess.run(["python", ASK_PY, ai, prompt], capture_output=True, encoding="utf-8", timeout=600,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))  # 黒い窓を出さない
        if p.returncode == 0:
            break
        print(f"ask.py {ai} 失敗 {attempt + 1}/3: {p.stderr[-200:]}", flush=True)
    else:
        raise RuntimeError(f"ask.py {ai} 失敗: {p.stderr[-300:]}")
    last = p.stderr.strip().splitlines()[-1]
    m = re.search(r"model=(\S+).*cost=\$([0-9.]+)", last)
    if not m:
        raise RuntimeError(f"ask.py の費用行を読めない: {last}")
    return p.stdout.strip(), m.group(1), float(m.group(2))


def ask_gemini(prompt):
    key = os.environ["GEMINI_API_KEY"]  # 環境変数から読む。無ければ KeyError で止まる
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODELS['gemini']}:generateContent"
    r = requests.post(url, headers={"x-goog-api-key": key}, json={
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0}}, timeout=120)
    r.raise_for_status()
    d = r.json()
    u = d["usageMetadata"]
    text = d["candidates"][0]["content"]["parts"][0]["text"]
    return text, u.get("promptTokenCount", 0), u.get("candidatesTokenCount", 0)


def ask_claude(prompt):
    key = os.environ["ANTHROPIC_API_KEY"]
    r = requests.post("https://api.anthropic.com/v1/messages", headers={
        "x-api-key": key, "anthropic-version": "2023-06-01"}, json={
        "model": MODELS["claude"], "max_tokens": 1024, "temperature": 0,
        "messages": [{"role": "user", "content": prompt}]}, timeout=120)
    r.raise_for_status()
    d = r.json()
    return d["content"][0]["text"], d["usage"]["input_tokens"], d["usage"]["output_tokens"]


ASKERS = {"qwen3-8b": ask_ollama, "qwen3-14b": lambda p: ask_ollama(p, "qwen3-14b"), "gemini": ask_gemini, "claude": ask_claude}

# 1M トークンあたりの単価(ドル) (入力, 出力)。ローカルは 0
PRICE = {"qwen3-8b": (0, 0), "qwen3-14b": (0, 0), "gemini": (0.10, 0.40), "claude": (1.0, 5.0)}


CLI_AIS = ("grok", "deepseek")


def answer(ai, question, chunks, calc=False):
    # 回答(根拠ページ付き)・かかった秒数・概算費用(ドル)・計算式の失敗一覧・モデル名・使われた断片の番号を返す
    t = time.time()
    errors = []
    if ai in CLI_AIS:
        try:
            text, model, cost = ask_cli(build_prompt(question, chunks, calc), ai)
        except RuntimeError as e:  # 3 回失敗した問は止めずに、失敗として記録する(不正解として判定される)
            return "(API が 3 回失敗)", time.time() - t, 0.0, [f"API失敗: {e}"], ai, []
        if calc:
            text, errors = apply_calc(text)
        text, used = attach_evidence(text.strip(), chunks)
        return text, time.time() - t, cost, errors, model, used
    text, tin, tout = ASKERS[ai](build_prompt(question, chunks, calc))
    if calc:
        text, errors = apply_calc(text)
    cost = (tin * PRICE[ai][0] + tout * PRICE[ai][1]) / 1e6
    text, used = attach_evidence(text.strip(), chunks)
    return text, time.time() - t, cost, errors, MODELS[ai], used
