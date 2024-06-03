from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunkDocs(doc, size):
    r_text_splitter = RecursiveCharacterTextSplitter(
        # Set custom chunk size
        chunk_size=size,
        chunk_overlap=0,
        separators=['\n\n', '\n', ' ', '']
    )
    split = r_text_splitter.split_documents(doc)
    # splits = r_text_splitter.split_text(doc)
    return split