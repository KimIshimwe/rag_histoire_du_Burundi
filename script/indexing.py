import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


load_dotenv()

def create_vector_store(chunks, db_path = "db_burundi/chroma_db"):
    # si le dossier n'existe pas Qwen/Qwen3-Embedding-4B
    if not os.path.exists(db_path):
        os.makedirs(db_path)
        
    embeddings = HuggingFaceEmbeddings(model_name = "Qwen/Qwen3-Embedding-0.6B",
                                       model_kwargs={'device': 'cpu'}
                                       )
    
    # création  et remplissage de la bd vectorielle
    vector_store = Chroma.from_documents(
        documents = chunks,
        embedding = embeddings,
        persist_directory = db_path
    )
    
    return vector_store

# test
if __name__ == "__main__":
    
    from ingestion import ingest_pdf
    path_to_pdf = "data"
    
    if os.path.exists(path_to_pdf):
        my_chunks = ingest_pdf(path_to_pdf)
        create_vector_store(my_chunks)
    else:
        print("Le PDF  n'est pas à l'endroit spécifié")