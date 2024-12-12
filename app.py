import streamlit as st
import requests
from src.crew import HealthChatbot

# Tiêu đề ứng dụng
st.title("Chatbot health children 🚑🤖")

# Hiển thị lịch sử hội thoại
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị tin nhắn trong lịch sử hội thoại
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Hàm gửi truy vấn đến API
def query_api(query):
    url = "http://127.0.0.1:8000/search/"  # Endpoint for FastAPI
    response = requests.post(url, json={"query": query})
    if response.status_code == 200:
        query = response.json().get("query", [])
        # words_list = [res['text'] for res in results]
        # topic_string = " ".join(words_list)
        result = {
            "topic": query
        }
        result_crew = HealthChatbot().crew().kickoff(inputs = result)
        if result_crew:
            return result_crew
        return "Không tìm thấy thông tin phù hợp."
    return "Lỗi khi kết nối đến API."

# Hộp chat người dùng nhập vào
user_input = st.chat_input("Nhập câu hỏi của bạn nha...")
if user_input:
    # Hiển thị tin nhắn người dùng  
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Gửi truy vấn đến API và nhận kết quả
    with st.spinner("Đang tìm câu trả lời..."):
        bot_response = query_api(user_input)
    
    # Hiển thị phản hồi từ chatbot
    st.chat_message("assistant").markdown(bot_response)
    st.session_state.messages.append({"role": "assistant", "content": bot_response})