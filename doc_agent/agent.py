import re
from model import Model 
from doc_parser import DocHandler
from prompt_templates import (
    CREATE_TODO_TEMPALTE, 
    REFINE_RESPONSE_TEMPLATE, 
    SUMMARIZE_RESPONSE_TEMPLATE
)
from dataclasses import dataclass
from typing import Any
from typing import Optional


@dataclass 
class Task: 
    question: str 
    response_from: Optional[int]

class DocAgent: 
    def __init__(self, model: Model, doc_handler: DocHandler): 
        self.llm = model 
        self.doc_handler = doc_handler
        self.todo = []
        self.results = []
        self.steps_taken = 0

    def _reset(self) -> None: 
        self.todo = [] 
        self.steps_taken = 0 

    def _is_completed(self) -> bool: 
        if self.steps_taken > 0 and len(self.todo) == 0: 
            self._reset()
            return True 
        return False

    def _requires_response(self, question: str): 
        pattern = r'\[(.*?)\]'
        match = re.search(pattern, question)
        if match: 
            return match.group(1)
        return None
    
    def _has_numeric_prefix(self, question: str) -> bool: 
        pattern = r'^\d+\.'
        return bool(re.match(pattern, question))
        
    def _create_todo(self, prompt: str) -> None: 
        todo_prompt = CREATE_TODO_TEMPALTE.format(prompt=prompt)
        response = self.llm(todo_prompt)

        # TODO: remove this later
        print(response)

        tasks = response.split("\n")
        for t in tasks: 
            if not self._has_numeric_prefix(t): 
                continue 

            question = tuple(t.split(maxsplit=1))[1]
            response_from = self._requires_response(question)
            if response_from: 
                response_from = int(response_from.split(" ")[-1])
            self.todo.append(Task(question, response_from))

    def _attempt_task(self, task):
        question = task.question 
        if task.response_from: 
            pass 

        citations = self.doc_handler.search(question)
        refined_question = self.doc_handler.ground_query(question, citations)
        response = self.llm(refined_question)

        refine_response_prompt = REFINE_RESPONSE_TEMPLATE.format(response=response)
        refined_response = self.llm(refine_response_prompt)

        self.results.append(refined_response)

        # verify TODO: remove later
        print(response)

    def run(self, prompt: str) -> str: 
        while not self._is_completed(): 
            self.step(prompt) 

        summarize_response_prompt = SUMMARIZE_RESPONSE_TEMPLATE.format()
        response = self.llm(summarize_response_prompt)

        print("final response: ", response)
        return response 
        
    def step(self, prompt: str) -> None: 
        if self.steps_taken == 0 and len(self.todo) == 0: 
            self._create_todo(prompt) 
        elif len(self.todo) > 0:
            task = self.todo.pop(0)
            self._attempt_task(task)
        else: 
            # pass?
            pass
        
        self.steps_taken += 1
        
    
if __name__ == "__main__": 
    model = Model("gpt-4o")
    da = DocAgent(model)

    prompt = "Who subject is taught 4th period on Tueday and what is its syllabus and who teaches it and what is their education level?"

    da._create_todo(prompt)