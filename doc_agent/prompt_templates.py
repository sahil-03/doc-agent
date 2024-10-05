SYSTEM_PROMPT = f"""
You are a helpful assistant chat bot that answers user queries accurately.
You must ground your answers in evidence that you have found from your
search, and you must address all parts of the question asked. DO NOT hallucinate. 
"""

QUERY_PROMPT_TEMPLATE = """
Answer the question based on on the information provided in the context below. 
Do not use any external knowledge. Be concise and to the point. 

Context: 
{context}
{generate_code}
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

GENERATE_SUMMARY_TEMPLATE = """
Given the contents of a document, provide a brief, helpful summary explaining what the document is about. 
Do not use any external knowledge. Be concise and to the point. 

Content: {content}

Summary: 
"""

SELECT_DOCUMENTS_TEMPLATE = """
Given the following documents and their summaries along with a prompt from the user, select the most relevant
documents that you need to answer the question. You may select more than one document. Just output the document numbers that
you need to us. If none of the documents can be used, output -1. Do not use any external knoweledge.

Output format (example): 
Document #1
Document #2

All documents: 

{doc_descriptions}

Question: {question}

Answer: 
"""

GENERATE_CODE_TEMPLATE = """

This question requires information from a CSV file. Using the CSV path(s) provided and the context form the text files, generate correct Python code in order to answer the question provided. 
print the last line of your code. 
"""

DATAFRAME_INFO_TEMPLATE = """
The dataframe path is: {path}
The first few rows of the dataframe are as follows: 
{df_content}
"""

CORRECT_CODE_TEMPLATE = """
There has been an error in the previous implementation. 

The following is the original query prompt: 
{original_query}

The previous (buggy) implementation is the following: 
{current_code}

The error is the following: 
{error}

Please edit the code in order the resolve the bug. 

Answer:
"""