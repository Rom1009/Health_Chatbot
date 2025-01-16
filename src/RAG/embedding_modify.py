from typing import Any, List
from InstructorEmbedding import INSTRUCTOR

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.embeddings import BaseEmbedding
import asyncio
from llama_index.core import Settings
from sentence_transformers import SentenceTransformer

class InstructorEmbeddings(BaseEmbedding):
    _model: INSTRUCTOR = PrivateAttr()
    _instruction: str = PrivateAttr()

    def __init__(
        self,
        instructor_model_name: str = "meandyou200175/phobert-finetune", 
        instruction: str = "Represent a document for semantic search: ",
        **kwargs: Any,
    ) -> None:

        super().__init__(**kwargs)
        self._model = SentenceTransformer(instructor_model_name)
        self._instruction = instruction

    @classmethod
    def class_name(cls) -> str:
        return "instructor"

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)
    
    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._aget_text_embedding(text)

    def _get_query_embedding(self, query: str) -> List[float]:
        embeddings = self._model.encode([[self._instruction, query]])
        return embeddings[0]
    
    def _get_text_embedding(self, text: str) -> List[float]:
        embeddings = self._model.encode([[self._instruction, text]])
        return embeddings[0]
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._model.encode(
            [[self._instruction, text] for text in texts]
        )
        return embeddings

if __name__ == "__main__":
    embed_model = InstructorEmbeddings(embed_batch_size = 2)
    Settings.embed_model = embed_model