import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Đọc file dữ liệu
file_path = "../../public/dataset/benhviennhi_articles.csv"
df = pd.read_csv(file_path)

# Chia nhỏ văn bản thành các chunks
def chunk_text(text, chunk_size=200):
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# Áp dụng chia nhỏ văn bản
chunks = []
for idx, row in df.iterrows():
    chunked_texts = chunk_text(row['Text'])
    for chunk in chunked_texts:
        chunks.append({"title": row["Title"], "text": chunk})

# Chuyển dữ liệu chunks thành DataFrame
chunks_df = pd.DataFrame(chunks)

# print(chunks_df.head())

# Tải mô hình embedding
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Tạo embedding cho mỗi chunk
embeddings = model.encode(chunks_df['text'].tolist(), show_progress_bar=True)
embeddings = np.array(embeddings).astype('float32')

# Tạo và lưu chỉ mục FAISS
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Lưu FAISS index và chunks để tái sử dụng
faiss.write_index(index, "faiss_index.bin")
chunks_df.to_csv("chunks_data.csv", index=False)
