import fitz 
import ollama
from langchain_core.documents import Document
from docling.document_converter import DocumentConverter
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

load_dotenv()

#DATA_DIR = "data/"
IMAGE_DIR = "data/extracted_images"

if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)
    
def get_image_description(image_path):
    try:
        res = ollama.chat(
            model = 'llava',
            messages = [{'role':'user', 'content': 'Décris précisément ce graphique ou cette image technique.', 'image': [image_path]}]
        )
        return res['message']['content']
    
    except Exception as e:
        return f"Erreur description image: {e}"
    
def ingest_pdf(directory_path):
    converter = DocumentConverter()
    all_docs = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1500,
                                                   chunk_overlap = 200, 
                                                   separators= ["\n\n# ", "\n\n## ", "\n\n", "\n", " "]
                                                    )
    # parcourir tous les pdf
    files = [f for f in os.listdir(directory_path) if f.endswith('.pdf')]
    
    for file_name in files:
        file_path = os.path.join(directory_path, file_name)
        
        # A. conversion du pdf en objet structuré puis en markdown pour les tableaux
        result = converter.convert(file_path)
        markdown_text = result.document.export_to_markdown()
        chunks = text_splitter.split_text(markdown_text)
        for chunk in chunks:
            all_docs.append(Document(page_content=chunk, metadata={"source":file_name, "type":"text"}))
            
    return all_docs
       
        
        
        
        
        
   