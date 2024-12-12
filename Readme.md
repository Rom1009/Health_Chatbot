# **1. Health Chatbot**

Health Chatbot là một ứng dụng hỗ trợ người dùng cung cấp thông tin sức khỏe thông qua giao diện đơn giản và dễ sử dụng. Dự án này sử dụng các công nghệ như `FastAPI`, `Streamlit`, và `crewAI`.

---

## **2. Yêu cầu hệ thống**

Trước khi bắt đầu, đảm bảo rằng hệ thống của bạn đáp ứng các yêu cầu sau:

- **Python**: Phiên bản `>=3.10, <=3.13`

Clone website 
```bash
    git clone https://github.com/Rom1009/Health_Chatbox.git
```

## **3. Khởi tạo environment**
```bash
    python -m venv venv
    python -m install requirements.txt
```

## **5. Chạy model**
Mở hai terminal: 
1. **Backend**
```bash
    cd src 
    cd apis
    uvicorn query_api:app --reload
```
2. **Frontend**
```bash
    streamlit run app.py
```



