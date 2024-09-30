import torch
import numpy as np 
from model import Model
from transformers import GPT2Tokenizer
from typing import Optional
from typing import Any


EMBEDDING_MODEL = ""


class Preprocessor: 

    @staticmethod 
    def pipeline(): 
        pass 


class Parser(Model): 
    def __init__(self, similarity_metric: Optional[str] = "cosine", **kwargs: Any):
        super().__init__(**kwargs)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        ) 
        self.docs = [] 
        self.vector_store = np.array([])
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        if hasattr(Parser, f"{similarity_metric}_similarity"): 
            self.similarity = getattr(Parser, f"{similarity_metric}_similarity")
        else: 
            raise ValueError(f"{similarity_metric} is NOT supported. Please choose 'cosine' or 'euclidean'.")

    @staticmethod 
    def cosine_similarity(vector_embd: np.ndarray, query_embd: np.ndarray) -> float:
        v_norm = np.linalg.norm(vector_embd, axis=1)
        q_norm = np.linalg.norm(query_embd)
        return np.dot(vector_embd, query_embd) / (v_norm * q_norm)

    @staticmethod 
    def euclidean_similarity(vector_embd: np.ndarray, query_embd: np.ndarray) -> float: 
        dists = np.linalg.norm(vector_embd - query_embd, axis=1)
        return 1 / (1 + dists)

    def add_file(self, file_name: str): 
        pass 

    def get_embedding_single(self, text_chunk: str) -> np.ndarray: 
        response = self.llm.embeddings.create(input=[text_chunk], model=EMBEDDING_MODEL)
        return response.data[0].embedding 
    
    def get_embedding_all(self, documents) -> np.ndarray:
        return np.array([self.get_embedding_single(doc) for doc in documents]) 
