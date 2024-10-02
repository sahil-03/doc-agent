import os
import gzip 
import pickle
import numpy as np 
from prompt_templates import (
    QUERY_PROMPT, 
    CONTEXT_TEMPLATE, 
    SEMANTIC_CONTEXT_TEMPLATE
)
from model import Model
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from typing import List
from typing import Any

@dataclass
class Document: 
    content: str 
    semantic_context: str 

@dataclass 
class Citation: 
    document: Document 
    similarity_score: float


EMBEDDING_MODEL = "text-embedding-ada-002"
DB_PATH = Path(__file__).parent / "vector_db_store.pickle"


class Preprocessor: 
    llm = Model(model_name="gpt-4o")

    @staticmethod 
    def _get_semantic_context(content_chunk: str, full_document: str) -> str: 
        query = SEMANTIC_CONTEXT_TEMPLATE.format(content=content_chunk, full_document=full_document)
        response = Preprocessor.llm(query) 
        return response

    @staticmethod 
    def _handle_csv(contents: str):
        docs = []
        rows = contents.split("\n")
        col_names = rows[0].split(",")
        for r in rows[1:]: 
            vals = r.split(",")
            content = ", ".join(f"{col_names[i]}: {vals[i]}" for i in range(len(col_names)))
            semantic_context = Preprocessor._get_semantic_context(content, contents)
            docs.append(Document(content, semantic_context))
        return docs 

    @staticmethod 
    def _handle_txt(contents: str): 
        pass

    @staticmethod 
    def preprocess_text(contents: str, file_type: str) -> List[Document]: 
        if file_type == "csv": 
            return Preprocessor._handle_csv(contents)
        elif file_type == "txt": 
            return Preprocessor._handle_txt(contents)  
        else: 
            raise ValueError(f"'{file_type}' is not currently supported.")

class DocHandler: 
    def __init__(self, model: Model, similarity_metric: Optional[str] = "cosine"):
        self.llm = model.llm
        self.documents = [] 
        self.vector_store = np.array([])
        if hasattr(DocHandler, f"{similarity_metric}_similarity"): 
            self.similarity_fn = getattr(DocHandler, f"{similarity_metric}_similarity")
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

    def get_embedding_single(self, text_chunk: str) -> np.ndarray: 
        response = self.llm.embeddings.create(input=[text_chunk], model=EMBEDDING_MODEL)
        return np.array(response.data[0].embedding) 
    
    def get_embedding_all(self, documents) -> np.ndarray:
        return np.array([self.get_embedding_single(doc) for doc in documents]) 

    def load(self): 
        with gzip.open(DB_PATH, "rb") as f: 
            data = pickle.load(f)
            self.documents = data["documents"] 
            self.documents = data["vector_store"]

    def save(self):
        with gzip.open(DB_PATH, "wb") as f: 
            updated_data = {"documents": self.documents, "vector_store": self.vector_store}
            pickle.dump(updated_data, f)

    def add_file(self, path: str): 
        assert os.path.isfile(path), f"The path provided ({path}) could not be found."

        contents = open(path, "r").read() 
        preprocessed_docs = Preprocessor.preprocess_text(contents, path.split(".")[-1])
      
        self.documents.extend([doc for doc in preprocessed_docs])
        embeddings = np.vstack([
            self.get_embedding_single(doc.content) for doc in preprocessed_docs
        ])
        if self.vector_store.size == 0: 
            self.vector_store = embeddings 
        else: 
            self.vector_store = np.vstack((self.vector_store, embeddings))
        
    def search(self, query: str, top_k: int = 5) -> List[Citation]: 
        query_embd = self.get_embedding_single(query)
        similarity_scores = self.similarity_fn(self.vector_store, query_embd)
        top_inds = np.argsort(similarity_scores)[::-1]

        citations = []
        for ind in top_inds:
            citations.append(Citation(self.documents[ind], similarity_scores[ind]))
            if len(citations) >= top_k: 
                break 
        return citations

    def ground_query(self, query: str, citations: List[Citation]) -> str:
        context = "" 
        for i, citation in enumerate(citations): 
            context += CONTEXT_TEMPLATE.format(
                index=i+1, content=citation.document.content, semantic_context=citation.document.semantic_context
            )
        return QUERY_PROMPT.format(context=context, prompt=query)




if __name__ == "__main__":
    m = Model(model_name="gpt-4o") 
    parser = DocHandler(m)
    
    parser.add_file("/Users/sahil/Documents/doc_agent/doc_agent/data/Grades.csv")

    print(len(parser.documents))
    print(parser.vector_store.size)

    prompt = "what is the capital of Australia?"
    citations = parser.search(prompt)
    new_prompt = parser.ground_query(prompt, citations)
    print(new_prompt)

    print(m(new_prompt))