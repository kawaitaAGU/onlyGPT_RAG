# 📁 構成: GitHub レポジトリに以下3つのファイルを置く構成
# - app.py             : Streamlit フロントエンド
# - sample.csv         : 類似問題検索のベース（事前アップロード）
# - requirements.txt   : ライブラリ定義

# ✅ app.py
import streamlit as st
import pandas as pd
from openai import OpenAI
from PIL import Image
import base64
import io
from pathlib import Path

# --- API キー取得 ---
if "OPENAI_API_KEY" not in st.secrets:
    st.error("OPENAI_API_KEY が設定されていません。")
    st.stop()

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --- CSVファイル読み込み ---
csv_path = Path("sample.csv")
df = pd.read_csv(csv_path)

# --- Streamlit UI ---
st.set_page_config(page_title="国家試験 類似問題GPT検索", layout="wide")
st.title("\U0001F4DA 国家試験問題 類似検索＋新作類題生成（GPTのみで検索）")

uploaded_file = st.file_uploader("国家試験の問題画像をアップロード", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="アップロード画像", use_column_width=True)

    with st.spinner("GPTが問題文を読み取り中..."):
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='PNG')
        image_bytes.seek(0)
        base64_image = base64.b64encode(image_bytes.read()).decode()

        # 画像から問題文を抽出
        vision_response = client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[
                {"role": "system", "content": "あなたは国家試験OCRエンジンです。画像から問題文・選択肢を正確に抽出してください。"},
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}]}
            ],
            max_tokens=1000
        )
        extracted_question = vision_response.choices[0].message.content.strip()
        st.markdown("### 抽出された問題文")
        st.markdown(extracted_question)

    with st.spinner("GPTが出題領域を判定中..."):
        domain_response = client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[
                {"role": "system", "content": "以下の国家試験問題は、どの出題領域（例：歯内療法、解剖、理工、生理、口腔外科など）に属しますか？1語で答えてください。"},
                {"role": "user", "content": extracted_question}
            ]
        )
        predicted_domain = domain_response.choices[0].message.content.strip()
        st.success(f"出題領域の推定: {predicted_domain}")

    with st.spinner("GPTがcsv内から類似問題10題を抽出中..."):
        sample_text = "\n\n".join([f"{i+1}. {row['設問']}\na. {row['選択肢a']}\nb. {row['選択肢b']}\nc. {row['選択肢c']}\nd. {row['選択肢d']}\ne. {row['選択肢e']}" for i, row in df[df['領域'].str.contains(predicted_domain, na=False)].head(20).iterrows()])

        similarity_response = client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[
                {"role": "system", "content": f"以下の国家試験問題リストの中から、次の問題に意味的に最も近い10問を選んでください。\n\n{sample_text}"},
                {"role": "user", "content": extracted_question}
            ],
            max_tokens=2000
        )
        similar_questions = similarity_response.choices[0].message.content.strip()
        st.markdown("### 類似問題（GPTが10問選出）")
        st.markdown(similar_questions)

    with st.spinner("GPTが解説と新作類題を生成中..."):
        final_response = client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[
                {"role": "system", "content": "あなたは歯科国家試験の教育AIです。抽出された問題と類似10問から、正解・解説・理由を記述し、さらに新作類題を3問生成してください。"},
                {"role": "user", "content": f"【未知の問題】\n{extracted_question}\n\n【類似問題】\n{similar_questions}"}
            ],
            max_tokens=2000
        )
        st.markdown("### GPTによる解説と類題")
        st.markdown(final_response.choices[0].message.content.strip())

# ✅ requirements.txt
# streamlit
# pandas
# openai
# pillow

# ✅ sample.csv
# 任意の国家試験問題データ。領域,設問,選択肢a,b,c,d,e,正解 を含む形式で用意してください。
