from llm import LLM

if __name__ == '__main__':
    llm = LLM()
    question = "what is star wars about?"
    answer = llm.generate_answer(question)
    print(answer)
