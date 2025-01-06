import streamlit as st
import requests
from src.crew import HealthChatbot
import os
from llama_index.core import  Settings,StorageContext, load_index_from_storage,VectorStoreIndex
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import json
import warnings
import random
import uuid
warnings.filterwarnings("ignore")

CHAT_HISTORY_FILE = "public/chat_history.json"

def vector_embediing():
    embed_model = HuggingFaceEmbedding(model_name= "BAAI/bge-small-en-v1.5")
    # embed_model = SentenceTransformer()
    return embed_model


# ... (Khởi tạo index và query_engine - Đảm bảo đã khởi tạo trước khi vào Streamlit)
Settings.llm = Groq(
    model="mixtral-8x7b-32768",
    api_key=os.environ.get("GROQ_API_KEY"),
)

Settings.embed_model = vector_embediing()

storage_context = StorageContext.from_defaults(
    persist_dir = "src/embedding/Test"
)

vector_index = load_index_from_storage(
    storage_context, 
)

query_engine= vector_index.as_query_engine(
    similarity_top_k = 3,
)

if query_engine is None:
    st.stop()

# Load/Save chat history
def load_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_chat_history(history):
    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=4)

def clear_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        os.remove(CHAT_HISTORY_FILE)
    st.session_state.chat_history = {}
    st.session_state.chat_id = None
    st.rerun()

def delete_chat_session(chat_id):
    if "chat_history" in st.session_state and chat_id in st.session_state.chat_history:
        del st.session_state.chat_history[chat_id]
        if len(st.session_state.chat_history) == 0:
            st.session_state.chat_id = None
        save_chat_history(st.session_state.chat_history)
        st.rerun()

def generate_related_questions(bot_response):
    keywords = [word for word in str(bot_response).split() if len(word) > 3]
    if not keywords:
        return []
    num_questions = min(3, len(keywords))
    related_questions = []
    for _ in range(num_questions):
        random_keyword_index = random.randint(10, min(20,len(keywords)))
        random_keyword = keywords[random_keyword_index]
        related_questions.append(f"Tôi muốn biết thêm về {random_keyword}.")

    while len(related_questions) < 3 and keywords:
        random_keyword_index = random.randint(0, min(20,len(keywords)))
        random_keyword = keywords[random_keyword_index]
        new_question = f"Bạn có thể giải thích thêm về {random_keyword} không?"
        if new_question not in related_questions:
            related_questions.append(new_question)
    while len(related_questions)<3:
        related_questions.append("Tôi có thể hỏi gì khác?")
    return related_questions[:3]

def get_or_create_chat_id():
    if "chat_id" not in st.session_state:
        st.session_state.chat_id = str(uuid.uuid4())
    return st.session_state.chat_id

def get_chat_messages(chat_id):
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = load_chat_history()
    if chat_id not in st.session_state.chat_history:
        st.session_state.chat_history[chat_id] = []
    return st.session_state.chat_history[chat_id]

def query_api(query, query_engine):
    """Gửi truy vấn đến API và trả về kết quả."""
    API_URL = "http://127.0.0.1:8000/search/"  # Thay thế bằng URL API của bạn

    try:
        response = requests.post(API_URL, json={"query": query})
        response.raise_for_status()  # Kiểm tra lỗi HTTP (4xx, 5xx)
        query_result = response.json().get("query", [])
        if query_result:
            ans = query_engine.query(query_result)
            print(ans)
            if ans:
                most_similar_question = ans.source_nodes[0].node.text if ans.source_nodes else None
                result = {"topic": most_similar_question}
                result_crew = HealthChatbot().crew().kickoff(inputs=result)

                if result_crew and query_engine:
                    related_queries = query_engine.query(query_result)
                    st.session_state.related_queries = [node.node.text for node in related_queries.source_nodes]
                    return result_crew
                elif not result_crew:
                    return "Không tìm thấy thông tin phù hợp."
                else:
                    return "Không có query engine"
            else:
                return "Không tìm thấy thông tin phù hợp."
        else:
            return "Không tìm thấy thông tin phù hợp."
    except requests.exceptions.RequestException as e:
        st.error(f"Lỗi kết nối API: {e}")
        return "Lỗi khi kết nối đến API."
    except json.JSONDecodeError as e:
        st.error(f"Lỗi giải mã JSON: {e}. Phản hồi từ API: {response.text}") #In ra phản hồi lỗi từ API để debug
        return "Lỗi khi xử lý dữ liệu từ API."
    except Exception as e:
        st.error(f"Lỗi trong quá trình xử lý: {e}")
        return "Lỗi trong quá trình xử lý."

# Khởi tạo giao diện
st.title("Chatbot health children")

# Initialize chat history in session state if it doesn't exist
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}

# Get or create chat ID
if "chat_id" not in st.session_state:
    st.session_state.chat_id = str(uuid.uuid4())
chat_id = st.session_state.chat_id

messages = st.session_state.chat_history.get(chat_id, [])

# Sidebar for chat history
with st.sidebar:
    st.header("Lịch sử chat")
    if st.button("Cuộc trò chuyện mới", use_container_width=True):
        st.session_state.chat_id = str(uuid.uuid4())
        st.session_state.messages = [] # Clear messages for new chat
        st.rerun()

    for chat_id_key, chat_messages in st.session_state.chat_history.items():
        if chat_messages:
            user_message = chat_messages[0]['content'][:50] if len(chat_messages) > 0 else ""
            bot_message = chat_messages[1]['content'][:50] if len(chat_messages) > 1 else ""
            st.write(f"**User:** {user_message}...")
            st.write(f"**Bot:** {bot_message}...")
            if st.button(f"Xóa phiên chat", key=f"delete_{chat_id_key}", use_container_width=True):
                delete_chat_session(chat_id_key)
                del st.session_state.chat_history[chat_id_key]
                st.rerun()

# Display chat messages
for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input and processing
if prompt := st.chat_input("Nhập câu hỏi của bạn nha..."):
    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Đang tìm câu trả lời..."):
        bot_response = query_api(prompt, query_engine)

    messages.append({"role": "assistant", "content": f"{bot_response}"})
    with st.chat_message("assistant"):
        st.markdown(bot_response)

    st.session_state.chat_history[chat_id] = messages
    save_chat_history(st.session_state.chat_history)
    st.rerun() # Rerun to display the new message

# Display related queries (outside the chat input block)
if messages: # Only show related queries if there are messages
    last_bot_response = messages[-1]["content"]
    related_queries = generate_related_questions(last_bot_response)
    if related_queries:
        st.write("Câu hỏi liên quan:")
        cols = st.columns(len(related_queries))
        for i, related_query in enumerate(related_queries):
            if cols[i].button(related_query, key=f"related_{i}"): # Unique keys are crucial
                st.session_state.messages = messages
                st.session_state.messages.append({"role": "user", "content": related_query})
                with st.chat_message("user"):
                    st.markdown(related_query)

                with st.spinner("Đang tìm câu trả lời..."):
                    bot_response = query_api(related_query, query_engine)

                st.session_state.messages.append({"role": "assistant", "content": bot_response})
                with st.chat_message("assistant"):
                    st.markdown(bot_response)
                st.session_state.chat_history[chat_id] = st.session_state.messages
                save_chat_history(st.session_state.chat_history)
                st.rerun()

# CSS (giữ nguyên)
st.markdown(
    """
    <style>
    .related-query-button {
        padding: 8px 16px;
        border: 1px solid #4CAF50;
        border-radius: 4px;
        background-color: #4CAF50;
        color: white;
        cursor: pointer;
        transition: background-color 0.3s;
        margin-bottom: 5px;
        width: fit-content;
        flex: 1 0 auto;
    }
    .related-query-button:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True,
)