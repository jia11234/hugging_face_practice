from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration, pipeline
import streamlit as st
import base64
import pandas as pd
from datetime import date

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

img_base64 = get_base64_image("images/grass.jpg")
img_tree1 = get_base64_image("images/tree1.png")
img_tree2 = get_base64_image("images/tree2.png")
img_tree3 = get_base64_image("images/tree3.png")
img_tree4 = get_base64_image("images/tree4.png")
img_tree5 = get_base64_image("images/tree5.png")
img_tree6 = get_base64_image("images/tree6.png")

score_df = pd.read_csv('data/data.csv')

if not score_df.empty:
    score = score_df['score'].iloc[-1]
else:
    score = 0

# 한 번 생성 후 재사용하기 위함
@st.cache_resource
# 감정 분류 모델 불러오기
# 파이프라인 생성 
def load_classifier():
    return pipeline(
        "text-classification",
        model="Jinuuuu/KoELECTRA_fine_tunning_emotion",
        tokenizer="Jinuuuu/KoELECTRA_fine_tunning_emotion"
    )
classifier = load_classifier()

# 문장 요약 모델 불러오기(task : SummSummarization)
def load_classifier():
    tokenizer = PreTrainedTokenizerFast.from_pretrained("EbanLee/kobart-summary-v3")
    model = BartForConditionalGeneration.from_pretrained("EbanLee/kobart-summary-v3")
    model.eval()
    return tokenizer, model

tokenizer, model = load_classifier()
#---css---
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@100;400;700&display=swap');
    .block-container {{
        padding-top: 100px;
        padding-bottom: 0rem;
        padding-left: 0rem;
        padding-right: 0rem;
    }}
    header {{
        visibility: hidden;
    }}
    html, body  {{
    font-family: 'Noto Sans KR', sans-serif;
    }}
    .tree_container {{
        display: flex;
        align-items: center;
        flex-direction: column;
        margin-bottom: 100px;
        margin-top: -80px;
    }}
    .trees{{
        height: 250px;
        display:flex;
        align-items: flex-end;
        justify-content:center;
        gap: 15px;
        max-width:1800px;
        color: #0E1117;
    }}
    .trees div{{
        width: 120px;
    }}
    .trees img{{
        width:100%
    }}
    .land{{
        width:100%
    }}
    .grass {{
        background-color: #12723B;
        height: 80px;
        border-radius: 10px 10px 0px 0px;
        background-image: url('data:image/jpg;base64,{img_base64}');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    .soil {{
        background-color: #51352B;
        height: 20px;
        margin-bottom: 10px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("나무를 키워보세요🌱")
st.markdown(
    """
    <div style="line-height: 1.8; margin-bottom: 10px; font-size: 18px;">
        다양한 감정이 느껴지는 일기를 작성해 보세요<br>
        감정이 풍부할수록 나무가 더 잘 자란답니다!
    </div>
    """, 
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <div class='tree_container'>
        """,
    unsafe_allow_html=True
)

# 점수별 나무 성장 단계 구현

# 트리 이미지 배열
tree_images = [
    img_tree1,
    img_tree2,
    img_tree3,
    img_tree4,
    img_tree5,
    img_tree6
]

# 40씩 나무가 한개씩 생김
tree_count = (score // 40) + 1      # 나무 개수
current_score = score % 40          # 마지막 나무의 자람 단계를 알기 위해 %40 해줌
grow= current_score//8              # 나무 자람 단계
current_tree_img = tree_images[grow]# 나무 자람 단계에 따른 이미지

# 나무를 여러개 출력시키기 위해 html 배열 생성
tree_html = []
for i in range(tree_count):
    if i == tree_count - 1:
        img = current_tree_img
    else:
        img = img_tree6

    tree_html.append(f"""
        <div><img src='data:image/jpg;base64,{img}'/></div>
    """)

# 나무들 출력
st.markdown(
    f"""
    <div class='trees'>
        {tree_html}
    </div>
    """,
    unsafe_allow_html=True
)
# 잔디 출력
st.markdown(
    f"""
        </div>
        <div class='land'>
            <div class='grass'>    
            </div>
            <div class='soil'>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

MAX_LEN = 300   # 최대 입력 길이

# 일기 작성 text_area
content = st.text_area(
    "일기 내용",
    placeholder="한 줄에 한 문장씩 적어보세요",
    height=150,
    max_chars=MAX_LEN
)

sentences = [] # 문장
labels=[] # 감정

emotion_dict = {
    "angry": "분노",
    "happy": "행복",
    "anxious": "불안",
    "embarrassed": "당황",
    "sad": "슬픔",
    "heartache": "상처"
}

# \n 기준으로 문장 나눔
if content.strip():
    sentences = [line.strip() for line in content.split("\n") if line.strip()]

result = " "

# 버튼을 누르면
if st.button("일기 작성 완료"):
    if content.strip():
        with st.spinner("당신의 일기를 정리 및 분석 중입니다."):

            # 일기 내용 토큰화
            inputs = tokenizer(
                content,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=512
            )

            # 문장 요약 모델을 이용하여 요약 결과 생성
            summary_text_ids = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                bos_token_id=model.config.bos_token_id,
                eos_token_id=model.config.eos_token_id,
                length_penalty=1.0,
                max_length=300,
                min_length=12,
                num_beams=6,
                repetition_penalty=1.5,
                no_repeat_ngram_size=15,
            )
            result = tokenizer.decode(summary_text_ids[0], skip_special_tokens=True)

            # 감정 분류 모델 예측 결과 처리
            result2 = classifier(sentences) 
            for item in result2:
                label = item["label"]
                if label not in labels:
                    labels.append(label)

        # 결과 출력 
            score += len(labels)*2
        labels = set(labels)
        kor = [emotion_dict.get(label, label) for label in labels]
        st.markdown(
            f"""
                <p style="font-family: 'Noto Sans KR'; color: #ffffff; font-size: 22px;">
                    "오늘의 기분 : {', '.join(kor)} "
                </p>
            <div style="
                background-color: #f0f4f2;
                padding: 20px;
                padding-bottom: 5px;
                border-left: 10px solid #5BA653;
                border-radius: 10px;
                margin-bottom: 20px;
            ">
                <p style="font-family: 'Noto Sans KR'; color: #000000; font-style: italic;">
                    " {result} "
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # 새로고침 시에도 score를 누적을 하기 위해 csv에 저장
        df = pd.DataFrame([{
            "date": date.today().isoformat(),
            "score": score,
            "emotion": ", ".join(labels),  
            "summary": " ".join(kor)    
        }])
        score_df = pd.concat([score_df, df], ignore_index=True)

        score_df.to_csv("data/data.csv", index=False)
    else:
        st.warning("내용을 입력해주세요.")
st.markdown(
    f"""
    <div style="height:20px;">
    </div>
    """,
    unsafe_allow_html=True
)