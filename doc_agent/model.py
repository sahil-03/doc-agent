import os
import openai
from prompt_templates import (
    SYSTEM_PROMPT, 
)
from typing import Optional
from typing import List 
from typing import Dict 

class Model: 
    def __init__(self, model_name: Optional[str] = "gpt-4o", api_key: Optional[str] = None): 
        self.model_name = model_name
        if api_key is not None:
            self.api_key = api_key
        else: 
            try:
                self.api_key = os.getenv("OPENAI_API_KEY")
            except ValueError as e: 
                raise ValueError(f"Please add OpenAI api key to your environment or provide one directly. Error: {e}")
        openai.api_key = self.api_key
        self.llm = openai.OpenAI()

    def __call__(self, prompt: str) -> str: 
        completion = self.llm.chat.completions.create(
            model=self.model_name,
            messages=self._format_message(prompt)
        )
        return completion.choices[0].message.content
    
    def _format_message(self, prompt: str) -> List[Dict[str, str]]: 
        return [
            {"role": "system", "content": SYSTEM_PROMPT}, 
            {"role": "user", "content": prompt}
        ]


# if __name__ == "__main__": 
#     m = Model() 
#     print(m("hello! what is your name"))