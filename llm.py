import os
from dotenv import load_dotenv
load_dotenv()

from langchain_upstage import ChatUpstage, UpstageEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}

def get_llm(model="solar-pro2"):
    return ChatUpstage(model=model)


def get_retriever():
    embeddings = UpstageEmbeddings(model="solar-embedding-1-large")
    vectorstore = PineconeVectorStore(
        index_name="pratice-langchain",
        embedding=embeddings
    )
    return vectorstore.as_retriever(search_kwargs={"k": 3})


def get_session_history(session_id: str) -> BaseChatMessageHistory:  
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def get_history_retriever():
    llm = get_llm()
    retriever = get_retriever()

    contextualize_q_system_prompt = (
        "대화 기록(chat history)과 최신 사용자 질문이 주어지면, "
        "대화 기록의 맥락을 참고해야 이해되는 질문일 수 있으므로 "
        "대화 기록 없이도 이해할 수 있는 독립적인 질문으로 재작성하세요. "
        "질문에 답변하지 말고, 필요한 경우에만 재작성하고, "
        "그렇지 않다면 원래 질문을 그대로 반환하세요."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    return create_history_aware_retriever(llm, retriever, contextualize_q_prompt)


def get_rag_chain():
    llm = get_llm()
    history_aware_retriever = get_history_retriever() 

    qa_system_prompt = """
당신은 공공데이터 ai 활용대회에 관한 내용을 알려주는 챗봇입니다.
질문에 답변할 때는 아래의 context를 활용하여 답변해주세요.
답변을 알 수 없다면 모른다고 답변해주세요.

context: {context}
"""
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    return conversational_rag_chain


def get_ai_response(user_message, session_id):
    rag_chain = get_rag_chain()
    ai_response = rag_chain.stream(
        {"input": user_message},
        config={"configurable": {"session_id": session_id}},
    )

    for chunk in ai_response:
        if "answer" in chunk:
            yield chunk["answer"]
            #깃허브 코드수정