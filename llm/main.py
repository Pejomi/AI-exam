from llm import LLM
from data_loader import loadYoutube, loadWiki, loadFile, loadDir
from pytube import Search
from utils import save_documents

if __name__ == '__main__':
    #save_documents()
    llm = LLM()
    question = "what to do in case of a traffic accident?"
    answer = llm.generate_answer(question)
    print(answer)
