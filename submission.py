import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  
import os

# ─── ページタイトル ───
st.title("予測結果の提出")

# ─── アップロード用ディレクトリ ───
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ─── Ground-truth の読み込み ───
GROUND_TRUTH_PATH = "XXX.csv"
try:
    ground_truth = pd.read_csv(GROUND_TRUTH_PATH)
except FileNotFoundError:
    st.error(f"Ground-truth ファイルが見つかりません：{GROUND_TRUTH_PATH}")
    st.stop()

# ─── 説明文 ───
st.markdown("""
## データセットの評価  
テストデータセット (predict_group(?).csv) に対する予測精度を Accuracy で評価します。

### 注意  
- ファイル名は predict_group(?).csv (? は班名) としてください。  
  - A班の場合: predict_groupA.csv  
  - オンラインの方々は、predict_group_氏名.csv  
- 予測結果は pred 列に格納してください。
""")

# ─── ファイルアップローダー ───
uploaded_file = st.file_uploader(
    "予測ファイルをアップロードしてください (CSV形式)", type="csv"
)
if uploaded_file is not None:
    dst = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(dst, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"ファイル {uploaded_file.name} がアップロードされました")

# ─── アップロード済みファイル一覧でリーダーボード作成 ───
files = sorted(os.listdir(UPLOAD_DIR))
leaderboard = []
for fn in files:
    path = os.path.join(UPLOAD_DIR, fn)
    try:
        df_pred = pd.read_csv(path)
        acc = (df_pred["pred"] == ground_truth["pred"]).mean()
        leaderboard.append({"ファイル名": fn, "Accuracy": acc})
    except Exception as e:
        # CSV 形式やカラムがおかしい場合は飛ばすorエラー表示
        st.warning(f"{fn} の読み込みに失敗: {e}")

if not leaderboard:
    st.warning("まだ提出がありません。提出ファイルをお待ちしています。")
    st.stop()

# ─── DataFrame にしてソート＆順位付け ───
lb = pd.DataFrame(leaderboard)
# Accuracy 列がない場合に備えてチェック
if "Accuracy" not in lb.columns:
    st.error(f"Accuracy 列が見つかりません。現在のカラム: {list(lb.columns)}")
    st.stop()

# ソート
lb = lb.sort_values("Accuracy", ascending=False).reset_index(drop=True)
# 順位列を付与
lb.index += 1
lb.insert(0, "順位", lb.index)

# メダル絵文字を付与
medals = {1: "🥇", 2: "🥈", 3: "🥉"}
lb["順位"] = lb["順位"].map(lambda i: f"{medals.get(i,'')} {i}" if i in medals else i)

# ─── 表示 ───
st.markdown("## リーダーボード")
st.dataframe(lb)