import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory

load_dotenv()
# initialisation des embeddings
def init_embeddings():
    return HuggingFaceEmbeddings(
        model_name = "Qwen/Qwen3-Embedding-0.6B",
        model_kwargs = {'device': 'cpu'}
        
    )

#initialisation de bd vectorielle
def init_db(embeddings):
    return Chroma(persist_directory= "db/chroma_db", embedding_function=embeddings)

#initialisation du LLM
def init_llm():
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
    
    llm_endpoint = HuggingFaceEndpoint(
        repo_id = repo_id,
        huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        temperature  = 0.5,
        max_new_tokens = 512,
        task = "text-generation",
    )

    return ChatHuggingFace(llm=llm_endpoint)

#génération de la réponse
def get_answer(query,db,llm,memory):
     # création du prompt
    template= """Utilise les extraits du document suivants pour répondre à la question à la fin.
    Si la réponse ne se trouve pas dans le contexte, dis que tu ne sais pas.
    
    Réponds en français et réponds uniquement à la question ne donne pas de détails.
    Sois factuel et utilise un ton professionnel.
    
    CONTEXTE: {context}
    QUESTION: {question}
    REPONSE: """
    
    prompt = PromptTemplate(template = template,input_variables = ["context", "question"])
    
    # création de la chaîne rag
    qa_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = db.as_retriever(search_kwargs = {"k":3}),
        chain_type_kwargs = {
            "prompt": prompt,
            "memory":memory
        }
        ,
        return_source_documents = False 
        )
    
    # testce qu'on parle de NDADAYEn

    #print(f"\n--- Question: {query} ---")
    response = qa_chain.invoke(query)
    return {
        "result": response["result"]
    }
    
   

    