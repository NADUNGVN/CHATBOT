import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration settings
class Config:
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
    EMBEDDING_MODEL_NAME = "dangvantuan/vietnamese-document-embedding"
    CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "chroma_pdf_db/Ho_Tich")
    PDF_DIRECTORY = os.getenv("PDF_DIRECTORY", "data/pdf")
    
    # Removed all security settings