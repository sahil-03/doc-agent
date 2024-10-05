import os
import openai
import instructor
from doc_agent.prompt_templates import SYSTEM_PROMPT
from pydantic import (
    BaseModel, 
    Field, 
    ConfigDict
)
from typing import Optional
from typing import List 
from typing import Dict 
from enum import Enum 


class ResultType(Enum): 
    DICT = "dict"
    DATAFRAME = "pandas dataframe"
    LIST = "list"
    SINGLE_ITEM = "single item"
    SERIES = "pandas series"


class PythonCodeGen(BaseModel): 
    python_code: Optional[str] = Field(None, description="The complete Python code required to fulfill the query.")
    expected_output_type: ResultType = Field(description = "The type of output expected from the query")
    python_libraries: List[str] = Field(default_factory=list, description="List of Python libraries required to execute the generated code.")
    dataframes_required: List[int] = Field(..., description="The list of dataframes denoted by their numbers required to be used in code for fulfilling the query")
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "python_code": "import pandas as pd\ndf = pd.read_csv('~/Documents/info.csv')\baverage_age = df['age'].mean()\nprint(average_age)",
                "expected_output_type": "single_item",
                "python_libraries": ["pandas"],
                "dataframes_required":[1],
            }
        }
    )


class DocumentSelection(BaseModel): 
    document_ids: List[int] = Field(description="A list of document indices that are relevant to answering the question.")
    reasoning: str = Field(description="A sentence or two explaining the reason behind selecting those documents.")
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "document_ids": [0, 1],
                "reasoning": "The question requires information that can be found in documents 0 and 1."
            }
        }
    )


class Model: 
    def __init__(self, model_name: Optional[str] = "gpt-4o", api_key: Optional[str] = None): 
        self.model_name = model_name
        if api_key is None:
            try:
                api_key = os.getenv("OPENAI_API_KEY")
            except ValueError as e: 
                raise ValueError(f"Please add OpenAI api key to your environment or provide one directly. Error: {e}")
        openai.api_key = api_key
        self.llm = instructor.patch(openai.OpenAI())

    def __call__(self, prompt: str, response_model: PythonCodeGen | DocumentSelection = None) -> str | List[int]: 
        if not response_model:
            completion = self.llm.chat.completions.create(
                model=self.model_name,
                messages=self._format_message(prompt),
            )
            return completion.choices[0].message.content
        
        completion = self.llm.chat.completions.create(
            model=self.model_name,
            messages=self._format_message(prompt),
            response_model=response_model
        )
        
        if hasattr(completion, "python_code"):
            return completion.python_code
        # print(completion.reasoning)
        return completion.document_ids
    
    def _format_message(self, prompt: str) -> List[Dict[str, str]]: 
        return [
            {"role": "system", "content": SYSTEM_PROMPT}, 
            {"role": "user", "content": prompt}
        ]
    