import argparse
from doc_agent.agent import DocAgent 
from doc_agent.model import Model 
from doc_agent.data_store import DataHandler, DataFile


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
