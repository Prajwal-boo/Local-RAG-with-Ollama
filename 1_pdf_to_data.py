from dotenv import load_dotenv
import os
import json
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

def pdf_to_jsonl(pdf_path, output_folder):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    
    print(f"Total pages loaded: {len(pages)}")
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    with open(os.path.join(output_folder, "data.txt"), "w", encoding="utf-8") as f:
        for page in pages:
            obj = {
                "url": f"local_pdf::genz_lingo::page_{page.metadata['page']}",
                "title": "GenZ Lingo Post 2023",
                "raw_text": page.page_content
            }
            f.write(json.dumps(obj) + "\n")
    
    print(f"data.txt created at {output_folder}")

pdf_to_jsonl(
    pdf_path=r"D:\RAG\Python Scripts\Local-RAG-with-Ollama\GenZ lingo.pdf",
    output_folder=os.getenv("DATASET_STORAGE_FOLDER")
)