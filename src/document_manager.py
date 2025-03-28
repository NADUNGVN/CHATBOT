import os
import shutil
from datetime import datetime
import tempfile
from langchain_community.document_loaders import PDFPlumberLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import pandas as pd

class DocumentManager:
    def __init__(self, pdf_directory, vector_directory):
        self.pdf_directory = pdf_directory
        self.vector_directory = vector_directory
        os.makedirs(pdf_directory, exist_ok=True)
    
    def get_document_list(self):
        """Return list of documents with metadata"""
        files = []
        for filename in os.listdir(self.pdf_directory):
            if filename.endswith('.pdf'):
                path = os.path.join(self.pdf_directory, filename)
                stat = os.stat(path)
                files.append({
                    "Tên tài liệu": filename,
                    "Ngày tải lên": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
                    "Kích thước": f"{stat.st_size / 1024:.1f} KB",
                    "Trạng thái": "Đã xử lý"
                })
        return pd.DataFrame(files)
    
    def add_documents(self, files):
        """Add new documents and create/update vector database"""
        try:
            os.makedirs(self.pdf_directory, exist_ok=True)
            os.makedirs(self.vector_directory, exist_ok=True)
            
            # Lưu files vào thư mục tạm
            with tempfile.TemporaryDirectory() as temp_dir:
                for file in files:
                    temp_path = os.path.join(temp_dir, os.path.basename(file.name))
                    with open(file.name, 'rb') as src, open(temp_path, 'wb') as dst:
                        dst.write(src.read())
                    final_path = os.path.join(self.pdf_directory, os.path.basename(file.name))
                    if os.path.exists(final_path):
                        os.remove(final_path)
                    shutil.copy2(temp_path, final_path)
            
            # Load và xử lý PDF files sử dụng DirectoryLoader với PDFPlumberLoader
            loader = DirectoryLoader(
                self.pdf_directory,
                glob="**/*.pdf",
                loader_cls=PDFPlumberLoader,  # Thay đổi loader class
                show_progress=True
            )
            documents = loader.load()
            
            if not documents:
                raise Exception("Không thể đọc tài liệu PDF")
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", " ", ""]
            )
            chunks = text_splitter.split_documents(documents)
            
            # Create vector database
            embeddings = HuggingFaceEmbeddings(
                model_name="dangvantuan/vietnamese-document-embedding",
                model_kwargs={"trust_remote_code": True}
            )
            
            # Phân loại và tạo vector database theo collections
            categories = ["trong_nuoc", "nuoc_ngoai", "dac_biet", "lien_thong"]
            for category in categories:
                category_dir = os.path.join(self.vector_directory, category)
                os.makedirs(category_dir, exist_ok=True)
            
            # Phân loại chunks theo category
            categorized_chunks = {category: [] for category in categories}
            for chunk in chunks:
                category = self._classify_document(chunk)
                categorized_chunks[category].append(chunk)
            
            # Tạo vector database cho từng category
            for category, category_chunks in categorized_chunks.items():
                if category_chunks:
                    category_dir = os.path.join(self.vector_directory, category)
                    Chroma.from_documents(
                        documents=category_chunks,
                        embedding=embeddings,
                        persist_directory=category_dir
                    )
            
            return True
            
        except Exception as e:
            raise Exception(f"Lỗi khi thêm tài liệu: {str(e)}")

    def _classify_document(self, document):
        """Phân loại tài liệu vào các collections"""
        source = document.metadata.get("source", "").lower()
        content = document.page_content.lower()
        
        if "nước ngoài" in source or "quốc tế" in content:
            return "nuoc_ngoai"
        elif "liên thông" in source or "liên thông" in content:
            return "lien_thong"
        elif any(term in source or term in content for term in 
                ["lưu động", "đăng ký lại", "đặc biệt"]):
            return "dac_biet"
        return "trong_nuoc"

    def delete_documents(self, selected_rows):
        """Delete documents and update vector database"""
        try:
            # Convert DataFrame row to dict if necessary
            if hasattr(selected_rows, 'to_dict'):
                selected_rows = selected_rows.to_dict('records')
            
            # Delete files with retry mechanism
            for row in selected_rows:
                if isinstance(row, dict):
                    filename = row.get("Tên tài liệu")
                else:
                    # If row is not a dict, skip it
                    continue
                    
                if not filename:
                    continue
                    
                filepath = os.path.join(self.pdf_directory, filename)
                if os.path.exists(filepath):
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            os.remove(filepath)
                            print(f"Đã xóa file: {filename}")
                            break
                        except PermissionError:
                            if attempt == max_retries - 1:
                                raise
                            import time
                            time.sleep(1)
            
            # Rebuild vector databases nếu còn files
            if os.listdir(self.pdf_directory):
                # Load lại PDF files sử dụng DirectoryLoader
                loader = DirectoryLoader(
                    self.pdf_directory,
                    glob="**/*.pdf",
                    loader_cls=PDFPlumberLoader,  # Thay đổi loader class
                    show_progress=True
                )
                documents = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                chunks = text_splitter.split_documents(documents)
                
                # Phân loại và tạo lại vector databases
                embeddings = HuggingFaceEmbeddings(
                    model_name="dangvantuan/vietnamese-document-embedding",
                    model_kwargs={"trust_remote_code": True}
                )
                
                categories = ["trong_nuoc", "nuoc_ngoai", "dac_biet", "lien_thong"]
                for category in categories:
                    category_dir = os.path.join(self.vector_directory, category)
                    if os.path.exists(category_dir):
                        shutil.rmtree(category_dir)
                    os.makedirs(category_dir, exist_ok=True)
                
                categorized_chunks = {category: [] for category in categories}
                for chunk in chunks:
                    category = self._classify_document(chunk)
                    categorized_chunks[category].append(chunk)
                
                for category, category_chunks in categorized_chunks.items():
                    if category_chunks:
                        category_dir = os.path.join(self.vector_directory, category)
                        Chroma.from_documents(
                            documents=category_chunks,
                            embedding=embeddings,
                            persist_directory=category_dir
                        )
            else:
                if os.path.exists(self.vector_directory):
                    shutil.rmtree(self.vector_directory)
                os.makedirs(self.vector_directory)
                print("Đã xóa toàn bộ vector database")
            return True
            
        except Exception as e:
            import traceback
            print(f"Error stack trace: {traceback.format_exc()}")
            raise Exception(f"Lỗi khi xóa tài liệu: {str(e)}")
