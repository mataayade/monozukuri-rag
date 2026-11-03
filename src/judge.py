# 判定役: 質問・正解・回答だけを外部 AI(glm、だめなら grok)に渡し、4 段階で判定させる
# 使い方  python src/judge.py <結果の名前> <AI名>   例: python src/judge.py 段3 qwen3-8b
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ASK = os.path.expanduser("~/.claude/scripts/ask.py")
LABELS = ("正解", "部分正解", "不正解", "でっち上げ")

PROMPT = """あなたは採点者です。「ものづくり白書」についての質問に対する回答を、正解と比べて判定してください。
判定の種類:
- 正解: 正解の要点(数値・名称・結論)が全て合っている。言い回しの違いや余分な説明は構わない
- 部分正解: 要点の一部だけ合っている、または要点が欠けている、別の事例・別の数値が混じっている
- 不正解: 要点が合っていない、または「資料には書かれていません」と答えたが正解には答えがある
- でっち上げ: 正解が「資料には書かれていない」なのに具体的な数値や事実を答えた、または正解と矛盾する具体的な数値・名称を作った
出力は次の 2 行だけ。他は書かない。
判定: <正解|部分正解|不正解|でっち上げ>
理由: <1 行>

【質問】{q}
【正解】{gold}
【回答】{ans}
"""


def ask_judge(prompt, who):
    # ask.py を呼んで出力を返す。失敗は例外にする
    p = subprocess.run(["python", ASK, who, prompt], capture_output=True, encoding="utf-8", timeout=300,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))  # 黒い窓を出さない
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-300:])
    return p.stdout


def parse(text):
    # 「判定: ○○」「理由: ○○」を取り出す。取れなければ None
    m = re.search(r"判定[:：]\s*(正解|部分正解|不正解|でっち上げ)", text)
    r = re.search(r"理由[:：]\s*(.+)", text)
    if not m:
        return None
    return m.group(1), (r.group(1).strip() if r else "")


def judge_one(row):
    prompt = PROMPT.format(q=row["question"], gold=row["gold"], ans=row["answer"])
    for who in ("glm", "glm", "grok"):  # glm を 2 回試し、だめなら grok
        try:
            res = parse(ask_judge(prompt, who))
        except RuntimeError as e:
            print("判定失敗", row["id"], who, e, flush=True)
            continue
        if res:
            return {"id": row["id"], "label": res[0], "reason": res[1], "judge": who}
    raise RuntimeError(f"判定できない: {row['id']}")


def main(tag, ai):
    rows = [json.loads(l) for l in open(f"results/{tag}_{ai}.jsonl", encoding="utf-8")]
    with ThreadPoolExecutor(4) as ex:
        out = list(ex.map(judge_one, rows))
    for o in out:
        o["ai"] = ai
    with open(f"results/判定_{tag}_{ai}.jsonl", "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(tag, ai, {l: sum(o["label"] == l for o in out) for l in LABELS}, "judge:", {w: sum(o["judge"] == w for o in out) for w in {o["judge"] for o in out}})


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
