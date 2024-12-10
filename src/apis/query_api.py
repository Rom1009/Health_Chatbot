from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

# Load FAISS index and data
index = faiss.read_index("../../public/dataset/faiss_index.bin")
chunks_df = pd.read_csv("../../public/dataset/chunks_data.csv")
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Initialize FastAPI
app = FastAPI()

# Define input and output schema
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class QueryResponse(BaseModel):
    query: str
    results: List[dict]

# Helper function to search FAISS index
def search_faiss_index(query: str, top_k: int):
    # Generate embedding for the query
    query_embedding = embedding_model.encode(query).astype(np.float32).reshape(1, -1)
    
    # Perform search
    distances, indices = index.search(query_embedding, top_k)
    
    # Fetch results from dataframe
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx != -1:  # Ensure valid index
            result = chunks_df.iloc[idx].to_dict()
            result["distance"] = float(dist)
            results.append(result)
    return results

# Define FastAPI endpoint
@app.post("/search", response_model=QueryResponse)
def search(query_request: QueryRequest):
    try:
        query = query_request.query
        top_k = query_request.top_k
        
        # Retrieve results
        results = search_faiss_index(query, top_k)
        if not results:
            raise HTTPException(status_code=404, detail="No relevant documents found.")
        
        return QueryResponse(query=query, results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


