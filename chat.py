import streamlit as st
from dotenv import load_dotenv
from llm import get_ai_response

load_dotenv()

st.set_page_config(
    page_title="공공태이터 ai활용대회 챗봇",
    page_icon="💩",
)

st.title("🤖")
st.caption("공공데이터 ai활용대회 챗봇입니다. 질문을 입력해주세요.")



# =====================
# session_id 생성 (중요)
# =====================
if "session_id" not in st.session_state:
    st.session_state.session_id = "session_" + str(id(st.session_state))

# =====================
# 메시지 상태
# =====================
if "messages_list" not in st.session_state:
    st.session_state.messages_list = []

# =====================
# 기존 메시지 출력
# =====================
for message in st.session_state.messages_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# =====================
# 사용자 입력
# =====================
if user_question := st.chat_input("ai 공공데이터 분석 챗봇"):
    with st.chat_message("user"):
        st.write(user_question)

    st.session_state.messages_list.append(
        {"role": "user", "content": user_question}
    )

    with st.spinner("AI가 답변을 작성하는 중..."):
        ai_response = get_ai_response(
            user_message=user_question,
            session_id=st.session_state.session_id
        )

        with st.chat_message("ai"):
            ai_message = st.write_stream(ai_response)

        st.session_state.messages_list.append(
            {"role": "ai", "content": ai_message}
        )
