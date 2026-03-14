import streamlit as st
import rag_chain as rag

st.set_page_config(page_title="DECOUVREZ L'HISTOIRE DU BURUNDI")

@st.cache_resource
def setup_system():

    emb = rag.init_embeddings()
    database = rag.init_db(emb)
    model = rag.init_llm()
    return database, model

db,llm = setup_system()

st.title("Assistant de découverte de l'histoire du pays au mille collines")

if query := st.chat_input("Votre question..."):
    with st.chat_message("user"):
        st.write(query)
        
    with st.chat_message("assistant"):
        with st.spinner("Génération de la réponse.."):
            output = rag.get_answer(query,db,llm)
            
            st.write(output["result"])