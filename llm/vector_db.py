from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

class VectorDB:
    def __init__(self):
        self.persist_directory = 'data/chroma/'
        self.embeddings = self.generate_embeddings()
    
    def generate_embeddings(self):
        model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        return embeddings
    
    def store_data(self, data):
        vector_db = Chroma.from_documents(
            documents=data,
            embeddings=self.embeddings,
            persist_directory=self.path
        )
        vector_db.persist()
        return vector_db
    
    def load_data(self):
        vector_db = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
        return vector_db