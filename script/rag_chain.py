import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA

load_dotenv()

def start_rag(query):
    
    embeddings = HuggingFaceEmbeddings(
        model_name = "Qwen/Qwen3-Embedding-0.6B",
        model_kwargs = {'device': 'cpu'}
        
    )
    
    # connexion db
    db  = Chroma(persist_directory= "db/chroma_db", embedding_function=embeddings)
    
    # config de l'IA
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
    
    llm = HuggingFaceEndpoint(
        repo_id = repo_id,
        huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        temperature  = 0.5,
        max_new_tokens = 512,
        task = "text-generation",
    )
    llm = ChatHuggingFace(llm=llm)
    
    # création du prompt
    template= """Utilise les extraits du document suivants pour répondre à la question à la fin.
    Si la réponse ne se trouve pas dans le contexte, dis que tu ne sais pas.
    
    Réponds en français et réponds uniquement ç la question ne donne pas de détails.
    
    CONTEXTE: {context}
    QUESTION: {question}
    REPONSE: """
    
    prompt = PromptTemplate(template = template,input_variables = ["context", "question"])
    
    # création de la chaîne rag
    qa_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = db.as_retriever(search_kwargs = {"k":3}),
        chain_type_kwargs = {"prompt": prompt},
        return_source_documents = True 
        )
    
    # testce qu'on parle de NDADAYEn

    #print(f"\n--- Question: {query} ---")
    response = qa_chain.invoke(query)
    #print("\n--- Réponse de l'IA ---")
    print(response["result"])
    
    #print("\n--- Sources: ---")
    sources = response["source_documents"]
    for i, doc in enumerate(sources):
        page = doc.metadata.get("page", "N/A")
       # print(f"Source {i+1}: Page {page}")
        
    
    
if __name__ == "__main__":
        start_rag("Quelle ethnie est majoritaire ?")
    