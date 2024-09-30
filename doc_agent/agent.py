from .model import Model 
from typing import Any

class DocAgent(Model): 
    def __init__(self, **kwargs: Any): 
        super().__init__(**kwargs)

    def _is_completed(self): 
        pass     

    def run(self): 
        while not self._is_completed(): 
            self.step()  

    def step(): 
        pass
    
