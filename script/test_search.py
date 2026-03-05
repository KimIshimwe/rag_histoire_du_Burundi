import langchain_huggingface
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def  test_search(query, db_path = "db/chroma_db"):
    embeddings = HuggingFaceEmbeddings(
        model_name =  "sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'})
    
    # connexion à la base
    db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    print(f"Nombre de documents dans la base : {db._collection.count()}")
    
    results = db.similarity_search(query, k=3)
    
    print(f"\n Resultat pour la question: '{query}' ")
    
    if len(results) == 0:
        print("Aucun document trouvé")
    else:
        
        for i, doc in enumerate(results):
            print(f"\n ---- Morceau {i+1} ----")
            print(doc.page_content[:900]+ "...")
            
    
if __name__ == "__main__":
    test_search("Un monarque impuissant face aux divisions politiques, claniques et ethniques")
