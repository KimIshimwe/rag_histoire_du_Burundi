from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

load_dotenv()

def ingest_pdf(path):
    loader = PyPDFLoader(path)
    pages = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200
    )
    chunks = text_splitter.split_documents(pages)
    
    return chunks

if __name__ == "__main__":
    pdf = "data/history_of_Burundi.pdf"
    if os.path.exists(pdf):
        docs = ingest_pdf(pdf)
        # premier morceau
        print(docs[0].page_content + "...")
    else:
        print("Le fichier PDF pas retrouvé.")