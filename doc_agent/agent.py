import io
import ast
import pandas as pd
from doc_agent.model import Model, PythonCodeGen, DocumentSelection
from doc_agent.data_store import DataHandler, DataFile
from doc_agent.prompt_templates import (
    CREATE_TODO_TEMPALTE, 
    SELECT_DOCUMENTS_TEMPLATE, 
    QUERY_PROMPT_TEMPLATE,
    GENERATE_CODE_TEMPLATE, 
    DATAFRAME_INFO_TEMPLATE,
    CORRECT_CODE_TEMPLATE
)
from typing import Any
from typing import Optional
from typing import List
from typing import Dict




class DocAgent: 
    def __init__(self, model: Model, data_handler: DataHandler, max_code_retries: int = 5): 
        self.llm = model 
        self.data_handler = data_handler
        self.max_retries = max_code_retries

    def _select_documents(self, prompt: str) -> List[int]:
        doc_descriptions = "\n".join(
            f"Document #{i}\nPath: {doc_file.path}\nSummary: {doc_file.summary}\n" for i, doc_file in enumerate(self.data_handler.data_store))
        select_docs_prompt = SELECT_DOCUMENTS_TEMPLATE.format(doc_descriptions=doc_descriptions, question=prompt)
        response = self.llm(select_docs_prompt, response_model=DocumentSelection)
        
        if -1 in response: 
            return []
        return response

    def _run_code(self, code: str) -> Dict[str, str]:
        from contextlib import redirect_stdout
        stdout_capture = io.StringIO()

        try: 
            tree = ast.parse(code)
            module = ast.Module(tree.body, type_ignores=[])
            code = compile(module, filename="<ast>", mode="exec")
            namespace = {'pd': pd}

            with redirect_stdout(stdout_capture):
                exec(code, namespace)

            # Capture any printed output
            printed_output = stdout_capture.getvalue().strip()
    
            if isinstance(tree.body[-1], ast.Expr):
                output = eval(compile(ast.Expression(tree.body[-1].value), filename="<ast>", mode="eval"), namespace)
                if printed_output: 
                    return {"output": f"Printed:\n{printed_output}\n\nEval:\n{output}"}
                return {"output": f"Eval:\n{output}"}
            elif printed_output: 
                return {"output": f"Printed:\n{printed_output}"}
            else: 
                return {"error": f"Error: The code executed successfully but no output was captured. Please output the result at the end of the code."}    
        except Exception as e: 
            return {"error": f"Error: {type(e).__name__}: {str(e)}"} 
    
    def _correct_code(self, original_query_prompt: str, current_code: str, error: str) -> str: 
        correct_code_prompt = CORRECT_CODE_TEMPLATE.format(
            original_query=original_query_prompt, current_code=current_code, error=error
        )
        response = self.llm(correct_code_prompt, response_model=PythonCodeGen)
        return response

    def _evaluate_code(self, original_query_prompt: str, code: str):
        print("Testing code...") 
        result = self._run_code(code)
        print(result)
        if "error" in result: 
            for retry in range(self.max_retries): 
                print(f"RETRY #{retry}")

                corrected_code = self._correct_code(original_query_prompt, code, result["error"])
                result = self._run_code(corrected_code)
                if "error" not in result:
                    break
        return result
        
    def _attempt_task(self, prompt: str, doc_inds: List[int]) -> None:
        context = ""
        generate_code = False
        for i in doc_inds: 
            if self.data_handler.data_store[i].content_type == "dataframe":
                context += f"\nDocument #{i}:\nContent:\n{DATAFRAME_INFO_TEMPLATE.format(path=self.data_handler.data_store[i].path, df_content=self.data_handler.data_store[i].content.head().to_string())}" 
                generate_code = True 
            else: 
                context += f"\nDocument #{i}:\nContent: {self.data_handler.data_store[i].content}\n"

        generate_code_prompt = "" if not generate_code else GENERATE_CODE_TEMPLATE
        query_prompt = QUERY_PROMPT_TEMPLATE.format(context=context, prompt=prompt, generate_code=generate_code_prompt)
        
        if generate_code: 
            print("Producing python code...")
        response = self.llm(query_prompt, response_model=PythonCodeGen if generate_code else None)
        if generate_code: 
            return self._evaluate_code(query_prompt, response)
        return response

    def run(self, prompt: str) -> str: 
        print("Looking through data...")
        doc_inds = self._select_documents(prompt)
        if len(doc_inds) == 0: 
            return "There is not enough relevant context to answer this question."
    
        print("Thinking...")
        response = self._attempt_task(prompt, doc_inds)

        if isinstance(response, dict) and "output" in response: 
            return response["output"]
        return response



    
if __name__ == "__main__": 
    model = Model("gpt-4o")
    doc_handler = DataHandler(model)
    agent = DocAgent(model, doc_handler)

    # prompt = "what is the name of the teacher who teaches 4th period on Wednesday and what is their age and summarize the syllabus of the class that they teach?"
    # prompt = "What is Ava Chen's age?"
    # prompt = "What is the syllabus for the subject taught in 3rd period on Monday?"
    # prompt = "What is the syllabus for chemistry?"
    # prompt = "Which subject has the lowest average grade across all students? Find its syllabus."
    # prompt = "What is the average grade for the subject taught in 1st period on Tuesday?"
    prompt = "What are the averages of each subject?"
    # prompt = "What is the 3rd element of the periodic table?"

    response = agent.run(prompt)
    print(response)
