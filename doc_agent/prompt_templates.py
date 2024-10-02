SYSTEM_PROMPT = f"""
You are a helpful assistant chat bot that answers user queries accurately.
You must ground your answers in evidence that you have found from your
search, and you must address all parts of the question asked. DO NOT hallucinate. 
"""

QUERY_PROMPT = """
Answer the question based on on the information provided in the context below. 
If the answer cannot be determined from the context, say "I do not have enough information to answer that question." 
Do not use any external knowledge. Be concise and to the point. 

Context: 
{context}

Question: {prompt}

Answer: 
"""

CONTEXT_TEMPLATE = """#{index}\nRaw Content: {content}\nSemantic Context: {semantic_context}\n\n"""

SEMANTIC_CONTEXT_TEMPLATE = """
Generate a one to two sentence summary of the raw content within the context of the full document. 
Do not use any external knowledge. Be concise and to the point. 

Raw Content: {content}

Full Document: {full_document}
"""

CREATE_TODO_TEMPALTE = """
Generate an enumerated list of sub-questions in order to answer the original questions. The output must be 
in a format of the example shown. If the question does not have more than one part, do not output anything
for the todo. Do not use any external knowledge. Be concise and to the point. 

Here is an example: 
Question: Who teaches Mathematics and what is their age? 

TODO: 
1. Who teaches Mathematics?
2. What is [response to 1]'s age?

Now, it is your turn. 

Question: {prompt}

TODO:
"""

REFINE_RESPONSE_TEMPLATE = """
Refine the following answer to the question by removing unecessary words that do not directly add to the 
answer. Do not use any external knowledge, Be concise and to the point. 

Here is an example: 
Question: What is the name of the teacher? 
Response: The teacher's name is John Smith. 

Refined response: John Smith 

Now, it is your turn. 

Question: {question}
Response: {response}

Refined response:
"""

SUMMARIZE_RESPONSE_TEMPLATE = """

"""