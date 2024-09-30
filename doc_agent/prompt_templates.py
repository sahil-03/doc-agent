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

