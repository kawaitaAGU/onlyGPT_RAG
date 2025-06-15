
import streamlit as st
st.set_page_config(page_title="国家試験 GPT検索", layout="wide")

import pandas as pd
from openai import OpenAI
from PIL import Image
import base64
import io
from pathlib import Path

if "OPENAI_API_KEY" not in st.secrets:
    st.error("OPENAI_API_KEY が設定されていません。")
    st.stop()

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

csv_path = Path("sample.csv")
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip().str.strip('"\'')

required_cols = ["設問", "選択肢a", "選択肢b", "選択肢c", "選択肢d", "選択肢e", "正解"]
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    st.error(f"次の列がCSVに見つかりませんでした: {missing_cols}")
    st.stop()

st.title("国家試験 類似検索＋新作類題生成（GPT検索のみ）")

uploaded_file = st.file_uploader("国家試験の問題画像をアップロード", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="アップロード画像", use_column_width=True)

    with st.spinner("GPTが問題文を読み取り中..."):
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='PNG')
        image_bytes.seek(0)
        base64_image = base64.b64encode(image_bytes.read()).decode()

        vision_response = client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[
                {"role": "system", "content": "あなたは国家試験OCRエンジンです。画像から問題文・選択肢・解答を正確に抽出してください。"},
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
                {"role": "system", "content": "この問題は歯科のどの出題領域（例：歯内療法、解剖、理工、生理など）に属しますか？1語で答えてください。"},
                {"role": "user", "content": extracted_question}
            ]
        )
        predicted_domain = domain_response.choices[0].message.content.strip()
        st.success(f"出題領域の推定: {predicted_domain}")

    with st.spinner("GPTがcsv内から類似問題10題を抽出中..."):
        sample_df = df.head(50)
        sample_text = "\n\n".join([
            f"{i+1}. {row['設問']}\na. {row['選択肢a']}\nb. {row['選択肢b']}\nc. {row['選択肢c']}\nd. {row['選択肢d']}\ne. {row['選択肢e']}"
            for i, row in sample_df.iterrows()
        ])

        similarity_response = client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[
                {"role": "system", "content": f"以下の国家試験問題リストは複数の領域にまたがっています。"
                                              f"あなたはこの中から、問題「{extracted_question}」が属すると推定された領域「{predicted_domain}」に関して、"
                                              "意味的に最も関連性の高い10問を選んでください。"},
                {"role": "user", "content": sample_text}
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
                {"role": "system", "content": "あなたは歯科国家試験の教育AIです。抽出された問題と類似10問から、正解・解説・理由を記述し、さらに新作類題を3問生成してください。その類題には解答と説明文を付け加えてください。"},
                {"role": "user", "content": f"【未知の問題】\n{extracted_question}\n\n【類似問題】\n{similar_questions}"}
            ],
            max_tokens=2000
        )
        st.markdown("### GPTによる解説と類題")
        st.markdown(final_response.choices[0].message.content.strip())
