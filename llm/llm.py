from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.chains import load_chain
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__)))
from vector_db import VectorDB
from chunking import chunkDocs

class LLM:
    def __init__(self):
        self.model = "mistral"
        self.db = self.get_vector_db()

    def get_vector_db(self):
        db = VectorDB()
        if db.load_data() is None:
            # TODO: insert our data into the db
            wiki = load_lib.loadWiki("some text", "en", 2)
            data = [wiki]
            chunk = chunkDocs(data, 350)
            db.store_data(chunk)
        return db

    def generate_answer(self, question):
        llm = Ollama(model=self.model, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

        # Build prompt
        template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer. 
        Use five sentences maximum. Keep the answer as concise as possible. 
        Always say "thanks for asking!" at the end of the answer. 

        {context}

        Question: {question}

        Helpful Answer:
        """
        prompt = PromptTemplate.from_template(template)
        
        chain = RetrievalQA.from_chain_type(
            llm,
            retriever=self.db.load_data().as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt})

        result = chain({"query": question})

        return result["result"]