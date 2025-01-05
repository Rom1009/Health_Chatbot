from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

# Initialize FastAPI
app = FastAPI()

# Define input and output schema
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class QueryResponse(BaseModel):
    query: str
    # results: List[dict]

# Define FastAPI endpoint
@app.post("/search", response_model=QueryResponse)
def search(query_request: QueryRequest):
    try:
        query = query_request.query
        # top_k = query_request.top_k
        
        # Retrieve results
        # results = search_faiss_index(query, top_k)
        # if not results:
            # raise HTTPException(status_code=404, detail="No relevant documents found.")
        
        return QueryResponse(query=query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))