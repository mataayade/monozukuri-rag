# 1 問だけ画面と同じ流れ(検索 → 回答)で流して、壊れていないか確かめる
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import os  # noqa: E402
os.chdir(ROOT)

from answer import answer  # noqa: E402
from search import Searcher  # noqa: E402

chunks = Searcher().search("マイスター・ハイスクールとはどんな事業ですか", k=8)
text, _, _, errors, model, used = answer("qwen3-14b", "マイスター・ハイスクールとはどんな事業ですか", chunks)
if errors or "マイスター" not in text or not used:
    print(f"[NG] errors={errors} used={used} text={text[:80]}")
    sys.exit(1)
print(f"[OK] {model} p.{chunks[used[0]]['paper_page']}")
