import streamlit as st
import pandas as pd
from openai import OpenAI
from PIL import Image
import base64
import io
from pathlib import Path

st.set_page_config(page_title="国家試験 GPT検索", layout="wide")
st.title("国家試験 類似検索＋新作類題生成（GPT検索のみ）")

# OpenAI APIキー
if "OPENAI_API_KEY" not in st.secrets:
    st.error("OPENAI_API_KEY が設定されていません。")
    st.stop()

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# CSV 読み込み
csv_path = Path("sample.csv")
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip().str.strip('"\'')

# 必要カラムの確認
required_cols = ["設問", "選択肢a", "選択肢b", "選択肢c", "選択肢d", "選択肢e", "正解"]
if not all(col in df.columns for col in required_cols):
    st.error(f"CSVに必要なカラムが不足しています: {required_cols}")
    st.stop()

# 画像アップロード
uploaded_file = st.file_uploader("国家試験の問題画像をアップロード", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="アップロード画像", use_column_width=True)

    # OCRによる問題文抽出
    with st.spinner("GPTが問題文を読み取り中..."):
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='PNG')
        base64_image = base64.b64encode(image_bytes.getvalue()).decode()

        vision_response = client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[
                {"role": "system", "content": "あなたは国家試験のOCR専門家です。画像から問題文・選択肢・正解を正確に抽出してください。"},
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}]}
            ],
            max_tokens=1000
        )
        extracted_question = vision_response.choices[0].message.content.strip()
        st.markdown("### 抽出された問題文")
        st.markdown(extracted_question)

    # 出題領域の分類
    with st.spinner("GPTが出題領域を判定中..."):
        domain_response = client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[
                {"role": "system", "content": "この問題は歯科のどの出題領域（例：理工、解剖、外科など）に分類されますか？1語で答えてください。"},
                {"role": "user", "content": extracted_question}
            ]
        )
        predicted_domain = domain_response.choices[0].message.content.strip()
        st.success(f"推定された出題領域: {predicted_domain}")

    # CSV全体を対象に類似検索
    with st.spinner("GPTがcsv内から類似問題10題を抽出中..."):
        full_text = "\n\n".join([
            f"{i+1}. {row['設問']}\na. {row['選択肢a']}\nb. {row['選択肢b']}\nc. {row['選択肢c']}\nd. {row['選択肢d']}\ne. {row['選択肢e']}"
            for i, row in df.iterrows()
        ])

        similarity_response = client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[
                {"role": "system", "content": f"次の国家試験問題リストには複数の領域の問題が混在しています。\n"
                                              f"あなたは問題「{extracted_question}」が属する「{predicted_domain}」領域に関する問題の中から、意味的に関連性の高いものを10問選んでください。"},
                {"role": "user", "content": full_text}
            ],
            max_tokens=4096
        )
        similar_questions = similarity_response.choices[0].message.content.strip()
        st.markdown("### 類似問題（GPTが10問選出）")
        st.markdown(similar_questions)

    # 解説と新作類題
    with st.spinner("GPTが解説と新作類題を生成中..."):
        final_response = client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[
                {"role": "system", "content": "あなたは歯科国家試験の教育AIです。抽出された問題と10問の類似問題に基づいて、"
                                              "正答・理由・解説を記述し、新作の類似問題を3問生成してください。それぞれに解答と解説をつけてください。"},
                {"role": "user", "content": f"【未知の問題】\n{extracted_question}\n\n【類似問題】\n{similar_questions}"}
            ],
            max_tokens=2000
        )
        st.markdown("### GPTによる解説と新作類題")
        st.markdown(final_response.choices[0].message.content.strip())
