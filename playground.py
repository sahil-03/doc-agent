import argparse
from doc_agent.agent import DocAgent 
from doc_agent.model import Model 
from doc_agent.data_store import DataHandler, DataFile


## SAMPLE QUESTIONS 
# prompt = "what is the name of the teacher who teaches 4th period on Wednesday and what is their age and summarize the syllabus of the class that they teach?"
# prompt = "What is Ava Chen's age?"
# prompt = "What is the syllabus for the subject taught in 3rd period on Monday?"
# prompt = "What is the syllabus for chemistry?"
# prompt = "Which subject has the lowest average grade across all students? Find its syllabus."
# prompt = "What is the average grade for the subject taught in 1st period on Tuesday?"
# prompt = "What are the averages of each subject?"
# prompt = "What is the 3rd element of the periodic table?"

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="DocAgent: An agent that can answer questions about your data.")
    
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--verbose", type=bool)
    parser.add_argument("--max_retries", type=int)

    args = parser.parse_args()

    model_name = args.model_name
    verbose = args.verbose
    max_retries = args.max_retries 

    model = Model(model_name)
    data_handler = DataHandler(model)
    agent = DocAgent(model, data_handler, max_code_retries=max_retries)

    print("Hi, I am DocAgent. Ask me any question about your data.")
    while True: 
        user_input = input("Enter a quesiton (or 'done' to exit): ").strip()
        if user_input.lower() == "done": 
            break 

        print("\nRESPONSE:\n")
        resp = agent.run(user_input)
        print(resp)


# prompt = "what is the name of the teacher who teaches 4th period on Wednesday and what is their age and summarize the syllabus of the class that they teach?"
# prompt = "What is Ava Chen's age?"
# prompt = "What is the syllabus for the subject taught in 3rd period on Monday?"
# prompt = "What is the syllabus for chemistry?"
# prompt = "Which subject has the lowest average grade across all students? Find its syllabus."
# prompt = "What is the average grade for the subject taught in 1st period on Tuesday?"
# prompt = "What are the averages of each subject?"
# prompt = "What is the 3rd element of the periodic table?"

# What is the syllabus for the subject taught in 3rd period on Monday?
# Which professors who teach mathematics are male?
# What is the average age of professors who teach Mathematics in the weekly schedule?
# What is the average grade for the subject taught in 1st period on Tuesday?
# What days and periods is the subject with the lowest average grade taught?
# What is the syllabus of the subject with the lowest average grade?