import streamlit as st
import rag_chain as rag
from langchain_community.chat_message_histories import StreamlitChatMessageHistory


st.set_page_config(page_title="DECOUVREZ L'HISTOIRE DU BURUNDI")
with st.sidebar:
    st.title("Paramètres")
    theme_choice = st.radio(
        "Confort visuel :",
        ["Mode Clair (Blanc)", "Mode Sombre (Noir)"],
        index=0
    )

 # personnalisation CSS
if theme_choice == "Mode Sombre (Noir)":
    st.markdown("""
        <style>
        .stApp {
            background-color: #0E1117;
            color: #FFFFFF;
        }
        /* Style pour les messages de chat en mode sombre */
        .stChatMessage {
            background-color: #262730;
        }
        </style>
        """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        .stApp {
            background-color: #FFFFFF;
            color: #000000;
        }
        </style>
        """, unsafe_allow_html=True)
@st.cache_resource
def setup_system():

    emb = rag.init_embeddings()
    database = rag.init_db(emb)
    model = rag.init_llm()
    return database, model

db,llm = setup_system()

#gestion historique
if "chat_history_text" not in st.session_state:
    st.session_state.chat_history_text = ""



# Interface du Chat 
if "messages" not in st.session_state:
    st.session_state.messages = [] 
    
st.title(" Posez moi des questions sur l'Histoire du Burundi !")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if query := st.chat_input("Posez votre question..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        # passer la mémoire stockée dans la session
        output = rag.get_answer(query, db, llm, st.session_state.chat_history_text)
        
        answer = output["result"]
        st.markdown(answer)
        
        # mise à jour historique
        st.session_state.chat_history_text += f"\nUtilisateur: {query}\nAssistant: {answer}\n"
        
    st.session_state.messages.append({"role": "assistant", "content": answer})