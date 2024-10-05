import os 
import pickle
import gzip 
import pandas as pd
from doc_agent.model import Model
from doc_agent.prompt_templates import GENERATE_SUMMARY_TEMPLATE
from dataclasses import dataclass 
from pathlib import Path

DB_PATH = Path(__file__).parent / "data_store.pkl.gz"

@dataclass 
class DataFile: 
    path: str 
    summary: str 
    content_type: str
    content: str | pd.DataFrame

class DataHandler: 
    def __init__(self, model: Model): 
        self.llm = model 
        self.data_store = []
        self._load_db()

    def _does_file_exist(self, path: str) -> bool: 
        for data_file in self.data_store:
            if path.split("/")[-1] == data_file.path.split("/")[-1]: 
                return True 
        return False
    
    def _save_db(self): 
        with gzip.open(DB_PATH, "wb") as f: 
            cur_data = {"data_store": self.data_store}
            pickle.dump(cur_data, f)

    def _load_db(self): 
        with gzip.open(DB_PATH, "rb") as f: 
            try: 
                data = pickle.load(f) 
            except EOFError: 
                return # File is empty 
            self.data_store = data["data_store"]
    
    def _get_summary(self, contents: str) -> str: 
        summary_prompt = GENERATE_SUMMARY_TEMPLATE.format(content=contents)
        response = self.llm(summary_prompt)
        return response
    
    def _handle_txt(self, path: str): 
        contents = open(path, "r").read() 
        summary = self._get_summary(contents)
        self.data_store.append(DataFile(path, summary, "raw_text", contents))

    def _handle_csv(self, path: str): 
        df = pd.read_csv(path)
        summary = self._get_summary(df.head())
        self.data_store.append(DataFile(path, summary, "dataframe", df)) 

    def add_file(self, path: str):
        if not os.path.isfile(path):
            print(f"{path} is not a valid path to a file.")
            return 
    
        if self._does_file_exist(path): 
            print("{path} already exists in our data store.")
            return

        file_type = path[-3:]
        if not hasattr(self, f"_handle_{file_type}"): 
            print("We do not support '{file_type}' file types.")
        getattr(self, f"_handle_{file_type}")(path)

        self._save_db()

if __name__ == "__main__": 
    m = Model("gpt-4o")
    dh = DataHandler(m)

    data_dir = "/Users/sahil/Documents/doc_agent/doc_agent/data"
    for file in os.listdir(data_dir): 
        path = os.path.join(data_dir, file)
        dh.add_file(path)