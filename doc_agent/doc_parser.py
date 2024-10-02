import os
import gzip 
import pickle
import nltk
import numpy as np 
from prompt_templates import (
    QUERY_PROMPT, 
    CONTEXT_TEMPLATE, 
    SEMANTIC_CONTEXT_TEMPLATE
)
from model import Model
from nltk.tokenize import sent_tokenize, word_tokenize
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from typing import List
from typing import Any

from concurrent.futures import ThreadPoolExecutor


nltk.download("punkt_tab", quiet=True)


EMBEDDING_MODEL = "text-embedding-ada-002"
DB_PATH = Path(__file__).parent / "vector_db.pkl.gz"


@dataclass
class Document: 
    content: str 
    semantic_context: str 


@dataclass 
class Citation: 
    document: Document 
    similarity_score: float


class Preprocessor: 
    llm = Model(model_name="gpt-4o-mini")

    @staticmethod 
    def _get_semantic_context(content_chunk: str, full_document: str) -> str: 
        query = SEMANTIC_CONTEXT_TEMPLATE.format(
            content=content_chunk, full_document=full_document
        )
        response = Preprocessor.llm(query) 
        return response

    @staticmethod 
    def _chunk_contents(contents: str, max_tokens: int = 500, overlap_tokens: int = 50) -> List[str]: 
        sentences = sent_tokenize(contents)
        chunks = []
        cur_chunk = ""
        num_tokens = 0

        for s in sentences: 
            tokens = word_tokenize(s)
            token_count = len(tokens)

            if num_tokens + token_count > max_tokens: 
                chunks.append(cur_chunk.strip())
                overlap = " ".join(word_tokenize(cur_chunk)[-overlap_tokens:])
                cur_chunk = overlap + " " + s
                num_tokens = len(word_tokenize(cur_chunk))
            else:
                cur_chunk += " " + s
                num_tokens += token_count
        
        if cur_chunk: 
            chunks.append(cur_chunk.strip())
        return chunks

    @staticmethod 
    def _handle_csv(contents: str):
        docs = []
        rows = contents.split("\n")
        col_names = rows[0].split(",")
        for r in rows[1:]: 
            vals = r.split(",")
            if len(vals) != len(col_names): 
                continue 

            content = ", ".join(f"{col_names[i]}: {vals[i]}" for i in range(len(col_names)))
            docs.append(Document(content, ""))
        return docs 

    @staticmethod 
    def _handle_txt(contents: str): 
        docs = []
        chunks = Preprocessor._chunk_contents(contents)
        for chunk in chunks: 
            semantic_context = Preprocessor._get_semantic_context(chunk, contents)
            docs.append(Document(chunk, semantic_context))
        return docs

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
        self.files = []
        self.vector_store = np.array([])
        if hasattr(DocHandler, f"{similarity_metric}_similarity"): 
            self.similarity_fn = getattr(DocHandler, f"{similarity_metric}_similarity")
        else: 
            raise ValueError(f"{similarity_metric} is NOT supported. Please choose 'cosine' or 'euclidean'.")

        # Load the vector db 
        self.load_db() 

    @staticmethod 
    def cosine_similarity(vector_embd: np.ndarray, query_embd: np.ndarray) -> float:
        v_norm = np.linalg.norm(vector_embd, axis=1)
        q_norm = np.linalg.norm(query_embd)
        return np.dot(vector_embd, query_embd) / (v_norm * q_norm)

    @staticmethod 
    def euclidean_similarity(vector_embd: np.ndarray, query_embd: np.ndarray) -> float: 
        dists = np.linalg.norm(vector_embd - query_embd, axis=1)
        return 1 / (1 + dists)

    def _is_file_processed(self, path: str) -> bool:
        filename = path.split("/")[-1]
        for f in self.files: 
            if filename == f.split("/")[-1]:
                return True
        return False

    def _group_docs(self, 
                    docs: List[Document], 
                    file_type: str, 
                    contents: str, 
                    doc_limit: int = 1000) -> List[Document]: 
        if len(docs) <= doc_limit: 
            return docs 

        i = 0
        num_docs = len(docs) // doc_limit
        cur_content = ""
        grouped_docs = []
        while i < len(docs): 
            if i > 0 and (i % num_docs == 0 or i == len(docs) - 1): 
                semantic_context = "" if file_type == "csv" else Preprocessor._get_semantic_context(cur_content, contents)
                grouped_docs.append(Document(cur_content, semantic_context))
                cur_content = ""
            else:
                cur_content += f" {docs[i].content}."
            i += 1
        return grouped_docs

    def get_embedding_single(self, text_chunk: str) -> np.ndarray: 
        response = self.llm.embeddings.create(input=[text_chunk], model=EMBEDDING_MODEL)
        return np.array(response.data[0].embedding) 
    
    def get_embedding_all(self, documents) -> np.ndarray:
        return np.array([self.get_embedding_single(doc) for doc in documents]) 

    def load_db(self): 
        with gzip.open(DB_PATH, "rb") as f: 
            try:
                data = pickle.load(f)
            except EOFError: 
                return  # File is empty
            self.documents = data["documents"]
            self.files = data["files"]
            self.vector_store = data["vector_store"]

    def save_db(self):
        with gzip.open(DB_PATH, "wb") as f: 
            updated_data = {"documents": self.documents, 
                            "files": self.files, 
                            "vector_store": self.vector_store}
            pickle.dump(updated_data, f)

    def add_file(self, path: str): 
        if not os.path.isfile(path): 
            print(f"The path provided ({path}) could not be found.")
            return
        
        if self._is_file_processed(path):
            print("This file has already been processed.")
            return

        contents = open(path, "r").read() 
        preprocessed_docs = Preprocessor.preprocess_text(contents, path.split(".")[-1])
        preprocessed_docs = self._group_docs(preprocessed_docs, path[-3:], contents)

        self.documents.extend(preprocessed_docs)
        embeddings = np.vstack([
            self.get_embedding_single(doc.content) for doc in preprocessed_docs
        ])

        if self.vector_store.size == 0: 
            self.vector_store = embeddings 
        else: 
            self.vector_store = np.vstack((self.vector_store, embeddings))
            
        self.files.append(path)
        self.save_db()
        
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
    model = Model(model_name="gpt-4o") 
    parser = DocHandler(model)

    data_dir = Path(__file__).parent / "data"
    for file in os.listdir(data_dir): 
        print("Processing: ", file)
        path_to_file = os.path.join(data_dir, file)
        parser.add_file(path_to_file)

    print(len(parser.files))
    