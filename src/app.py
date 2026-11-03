# Streamlit の画面: 質問 → 検索 → 回答と根拠を表示する(起動: streamlit run src/app.py)
import sys
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from answer import answer  # noqa: E402
from search import Searcher  # noqa: E402

# 画面の表示名 → answer.py の AI 名
AIS = {"qwen3:14b": "qwen3-14b", "qwen3:8b": "qwen3-8b", "grok": "grok", "deepseek": "deepseek"}


@st.cache_resource
def get_searcher():
    # データは起動時に読む(相対パスなので作業フォルダを固定する)
    import os
    os.chdir(ROOT)
    return Searcher()


st.title("ものづくり白書 質問窓口")
choice = st.selectbox("回答 AI", list(AIS))
with st.form("q", clear_on_submit=True):  # 送信したら入力欄を空にする
    question = st.text_input("質問")
    sent = st.form_submit_button("質問する")

history = st.session_state.setdefault("history", [])  # これまでの質問と回答(新しい順)

if sent and question.strip():
    st.markdown(f"**質問:** {question}")  # 待っている間も何を聞いたか見えるようにする
    searcher = get_searcher()
    t = time.time()
    with st.spinner("回答を作っています"):
        chunks = searcher.search(question, k=8)  # 段4で効いた k=8
        text, _, _, errors, model, used = answer(AIS[choice], question, chunks)  # 計算の Python 化は段4で不採用
    ordered = [chunks[i] for i in used] + [c for i, c in enumerate(chunks) if i not in used]  # 使われた断片を先頭に
    history.insert(0, {"question": question, "answer": text, "caption": f"{time.time() - t:.1f} 秒 / {model}",
                       "errors": errors, "chunks": ordered})
    # 質問と回答を記録する(あとで採点するため)
    import json
    with open(ROOT / "results" / "画面ログ.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": time.strftime("%H:%M:%S"), "ai": model, "question": question, "answer": text,
                            "pages": [chunks[i]["paper_page"] for i in used]}, ensure_ascii=False) + "\n")
    st.rerun()  # 一覧の表示に切り替える

# これまでの質問と回答をすべて表示する
for h in history:
    st.divider()
    st.markdown(f"**質問:** {h['question']}")
    st.write(h["answer"])
    st.caption(h["caption"])
    for e in h["errors"]:
        st.warning(e)
    with st.expander("根拠"):
        for c in h["chunks"]:
            st.markdown(f"**p.{c['paper_page']}**  {c['text'][:100]}")
