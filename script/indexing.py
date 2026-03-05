import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

def create_vector_store(chunks, db_path = "db/chroma_db"):
    # si le dossier n'existe pas 
    if not os.path.exists(db_path):
        os.makedirs(db_path)
        
    embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'}
                                       )
    
    # création de la bd vectorielle
    vector_store = Chroma.from_documents(
        documents = chunks,
        embedding = embeddings,
        persist_directory = db_path
    )
    
    return vector_store

# test
if __name__ == "__main__":
    
    from ingestion import ingest_pdf
    path_to_pdf = "data/history_of_Burundi.pdf"
    
    if os.path.exists(path_to_pdf):
        my_chunks = ingest_pdf(path_to_pdf)
        create_vector_store(my_chunks)
    else:
        print("Le PDF  n'est pas à l'endroit spécifié")