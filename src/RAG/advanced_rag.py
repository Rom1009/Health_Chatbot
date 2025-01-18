import os
import numpy as np
from llama_index.core import Document
from src.RAG.preprocessing import preprocessing
from llama_index.core import (
    VectorStoreIndex, 
    Settings, 
    StorageContext,
    load_index_from_storage,
    SimpleDirectoryReader
)
# from embedding_modify import InstructorEmbeddings
from llama_index.llms.groq import Groq
from transformers import AutoTokenizer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sentence_transformers import SentenceTransformer
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.indices.postprocessor import SentenceTransformerRerank, MetadataReplacementPostProcessor

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
def chunking(data, max_length=256):
    chunks = []
    for item in data:
        content = item["Content"]
        tokens = tokenizer.tokenize(content)
        
        for i in range(0, len(tokens), max_length):
            chunk_tokens = tokens[i:i + max_length]
            chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)
            
            # Lưu chunk cùng metadata
            chunks.append({
                "Tên bệnh": item["Chapter Title"],
                "Loại bệnh": item["Subheading Title"],
                "Chunk nội dung": chunk_text
            })
    return chunks

def vector_store_index(documents):
    vector_index = VectorStoreIndex.from_documents(
        documents=documents,
        show_progress=True,
    )
    vector_index.storage_context.persist(
        persist_dir= "../embedding/Advance_Rag"
    )
    return vector_index

def create_llama_documents(chunks):
    documents = []
    for item in chunks:
        metadata = {
            "Loại bệnh": item["Loại bệnh"],
            "Nội dung": item["Chunk nội dung"]
        }
        # embedding = np.array(item["Embedding"]) # Chuyển embedding thành numpy array (bắt buộc)
        document = Document(text=item["Chunk nội dung"], metadata=metadata) # text có thể rỗng
        documents.append(document)
    return documents

def embeddings(chunked_data):
    # Tạo embedding cho mỗi chunk
    model = SentenceTransformer("meandyou200175/phobert-finetune")
    embeddings = []
    for chunk in chunked_data:
        embedding = model.encode(chunk["Chunk nội dung"])
        embeddings.append({
            "Tên bệnh": chunk["Tên bệnh"],
            "Loại bệnh": chunk["Loại bệnh"],
            "Embedding": embedding
        })
    return embeddings

def get_sentence_window_query_engine(
    sentence_index,
    similarity_top_k=3,
    rerank_top_n=2,
):
    # define postprocessors
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="BAAI/bge-reranker-base"
    )

    sentence_window_engine = sentence_index.as_query_engine(
        similarity_top_k=similarity_top_k, node_postprocessors=[postproc, rerank]
    )
    return sentence_window_engine

def build_sentence_window_index(
    document, save_dir="sentence_index"
):
    # create the sentence window node parser w/ default settings
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    Settings.node_parser = node_parser
    if not os.path.exists(save_dir):
        sentence_index = VectorStoreIndex.from_documents(
            document
        )
        sentence_index.storage_context.persist(persist_dir=save_dir)
    else:
        sentence_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
        )

    return sentence_index    

if __name__ == "__main__":
    text_dict = preprocessing("../../public/document.pdf")
    # embed_model = InstructorEmbeddings(embed_batch_size = 2)
    embed_model = HuggingFaceEmbedding(model_name= "BAAI/bge-small-en-v1.5")
    Settings.embed_model = embed_model
    chunks = chunking(text_dict, max_length=512)
    # embedding = embeddings(chunks)
    documents = create_llama_documents(chunks)

    
    # index = vector_store_index(documents)
    Settings.llm = Groq(
        model="llama-3.3-70b-versatile",
        api_key=os.environ.get("GROQ_API_KEY"),
    )

    # sentence_index = build_sentence_window_index(
    #     document= documents,
    #     save_dir="../embedding/sentence_index"
    # )

    storage_context = StorageContext.from_defaults(
        persist_dir = "../embedding/sentence_index"
    )

    vector_index = load_index_from_storage(
        storage_context, 
    )
    sentence_window_engine = get_sentence_window_query_engine(vector_index)

    # query_engine = vector_index.as_query_engine(similar_top_k = 2)

    query = """
       Những biện pháp chăm sóc cần thiết tại phòng sinh cho trẻ sơ sinh là gì?
    """
    response = sentence_window_engine.query(query)
    print(response.source_nodes[0].get_text())
    print("---------------------------------------------------")
    print(response.response)