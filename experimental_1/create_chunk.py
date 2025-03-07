import os
import json
import logging
import fitz  # PyMuPDF
import re
from datetime import datetime
from typing import Dict, List
from pathlib import Path
from dataclasses import dataclass
import pdfplumber

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Cấu hình
BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = os.path.join(BASE_DIR, "data", "pdf")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "chunks")

# Cấu hình chunk
MIN_CHUNK_SIZE = 200
MAX_CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
MIN_SECTION_LENGTH = 50

# Các section cần gộp
MERGEABLE_SECTIONS = [
    "Lệ phí",
    "Phí",
    "Thời hạn giải quyết"
]

@dataclass
class TextChunk:
    content: str
    metadata: Dict
    
    def __post_init__(self):
        if not isinstance(self.metadata, dict):
            self.metadata = {}
        
        # Thêm timestamp
        self.metadata['created_at'] = datetime.now().isoformat()
        
        # Đảm bảo content là string
        if not isinstance(self.content, str):
            self.content = str(self.content)
        
        # Chuẩn hóa content
        self.content = self.content.strip()

def read_pdf_files(pdf_dir: str) -> Dict[str, str]:
    pdf_contents = {}

    for filename in os.listdir(pdf_dir):
        if filename.endswith('.pdf'):
            file_path = os.path.join(pdf_dir, filename)
            text = ""

            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""  # Trích xuất văn bản chính

                    # Xử lý bảng nếu có
                    tables = page.extract_tables()
                    for table in tables:
                        formatted_rows = []
                        for row in table:
                            # Chuyển None thành chuỗi rỗng ""
                            formatted_row = [cell if cell is not None else "" for cell in row]
                            formatted_rows.append(" | ".join(formatted_row))

                        table_text = "\n".join(formatted_rows)
                        text += f"\n---TABLE START---\n{table_text}\n---TABLE END---\n"

            pdf_contents[filename] = text

    return pdf_contents

class TextProcessor:
    def __init__(self):
        self.min_chunk_size = MIN_CHUNK_SIZE
        self.max_chunk_size = MAX_CHUNK_SIZE
        self.overlap_size = CHUNK_OVERLAP
        
    def extract_procedure_info(self, text: str) -> Dict:
        """Trích xuất thông tin thủ tục từ văn bản"""
        info = {
            'ma_thu_tuc': '',
            'ten_thu_tuc': '',
            'cap_thuc_hien': '',
            'linh_vuc': ''
        }
        
        # Tìm mã thủ tục
        ma_thu_tuc_match = re.search(r'Mã thủ tục:\s*([^\n]+)', text)
        if ma_thu_tuc_match:
            info['ma_thu_tuc'] = ma_thu_tuc_match.group(1).strip()
            
        # Tìm tên thủ tục
        ten_thu_tuc_match = re.search(r'Tên thủ tục:\s*([^\n]+)', text)
        if ten_thu_tuc_match:
            info['ten_thu_tuc'] = ten_thu_tuc_match.group(1).strip()
            
        # Tìm cấp thực hiện
        cap_match = re.search(r'Cấp thực hiện:\s*([^\n]+)', text)
        if cap_match:
            info['cap_thuc_hien'] = cap_match.group(1).strip()
            
        # Tìm lĩnh vực
        linh_vuc_match = re.search(r'Lĩnh vực:\s*([^\n]+)', text)
        if linh_vuc_match:
            info['linh_vuc'] = linh_vuc_match.group(1).strip()
            
        return info

    def split_into_sections(self, text: str) -> Dict[str, str]:
        sections = {}
        current_section = "Thông tin chung"
        current_content = []

        section_headers = [
            "Trình tự thực hiện",
            "Cách thức thực hiện",
            "Thành phần hồ sơ",
            "Thời hạn giải quyết",
            "Phí, lệ phí",
            "Căn cứ pháp lý",
            "Yêu cầu, điều kiện thực hiện"
        ]

        lines = text.split('\n')
        in_table = False
        table_data = []

        for line in lines:
            line = line.strip()

            if "---TABLE START---" in line:
                in_table = True
                table_data = []
                continue
            elif "---TABLE END---" in line:
                in_table = False
                sections[current_section] = "\n".join(table_data)
                continue
            
            if in_table:
                table_data.append(line)
                continue

            is_header = any(line.lower().startswith(header.lower()) for header in section_headers)

            if is_header:
                if current_content:
                    sections[current_section] = '\n'.join(current_content)
                current_section = line
                current_content = []
            else:
                current_content.append(line)

        if current_content:
            sections[current_section] = '\n'.join(current_content)

        return sections

    def split_into_paragraphs(self, text: str) -> List[str]:
        """Tách văn bản thành các đoạn có ý nghĩa"""
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]

    def split_long_paragraph(self, paragraph: str) -> List[str]:
        """Chia đoạn dài thành các chunk nhỏ hơn"""
        chunks = []
        words = paragraph.split()
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_len = len(word) + 1
            if current_length + word_len > self.max_chunk_size:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    overlap_words = current_chunk[-self.overlap_size:] if self.overlap_size < len(current_chunk) else current_chunk
                    current_chunk = overlap_words + [word]
                    current_length = sum(len(w) + 1 for w in current_chunk)
                else:
                    chunks.append(word)
                    current_chunk = []
                    current_length = 0
            else:
                current_chunk.append(word)
                current_length += word_len
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

    def create_chunk(self, content: str, metadata: Dict) -> Dict:
        """Tạo chunk mới với metadata"""
        return {
            "content": content.strip(),
            "metadata": metadata
        }

    def process_text(self, text: str, metadata: Dict) -> List[Dict]:
        procedure_info = self.extract_procedure_info(text)
        metadata.update(procedure_info)
        sections = self.split_into_sections(text)
        chunks = []
        
        for section_name, section_content in sections.items():
            paragraphs = self.split_into_paragraphs(section_content)
            current_chunk = []
            current_length = 0
            
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                    
                if len(para) > self.max_chunk_size:
                    if current_chunk:
                        chunk_content = "\n".join(current_chunk)
                        chunk_metadata = {
                            **metadata,
                            "section_name": section_name
                        }
                        chunks.append(self.create_chunk(chunk_content, chunk_metadata))
                        current_chunk = []
                        current_length = 0
                    
                    para_chunks = self.split_long_paragraph(para)
                    for p_chunk in para_chunks:
                        chunk_metadata = {
                            **metadata,
                            "section_name": section_name
                        }
                        chunks.append(self.create_chunk(p_chunk, chunk_metadata))
                else:
                    if current_length + len(para) > self.max_chunk_size:
                        chunk_content = "\n".join(current_chunk)
                        chunk_metadata = {
                            **metadata,
                            "section_name": section_name
                        }
                        chunks.append(self.create_chunk(chunk_content, chunk_metadata))
                        current_chunk = [para]
                        current_length = len(para)
                    else:
                        current_chunk.append(para)
                        current_length += len(para)
            
            if current_chunk:
                chunk_content = "\n".join(current_chunk)
                chunk_metadata = {
                    **metadata,
                    "section_name": section_name
                }
                chunks.append(self.create_chunk(chunk_content, chunk_metadata))
        
        return chunks

def save_chunks_to_json(chunks: List[TextChunk], output_dir: str, filename: str):
    """Lưu chunks vào file JSON riêng cho từng PDF"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        output_data = []
        for chunk in chunks:
            chunk_dict = {
                'content': chunk.content,
                'metadata': chunk.metadata
            }
            output_data.append(chunk_dict)
        
        # Tạo tên file dựa trên tên PDF gốc
        base_name = os.path.splitext(filename)[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"{base_name}_chunks_{timestamp}.json")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Đã lưu {len(chunks)} chunks từ file {filename} vào: {output_file}")
        return output_file
    
    except Exception as e:
        logger.error(f"Lỗi khi lưu chunks cho file {filename}: {str(e)}")
        raise

def main():
    try:
        # Đọc PDF files
        pdf_contents = read_pdf_files(PDF_DIR)
        if not pdf_contents:
            logger.error("Không tìm thấy file PDF nào để xử lý")
            return
        
        # Xử lý text
        processor = TextProcessor()
        
        # Xử lý từng file PDF riêng biệt
        for file_name, content in pdf_contents.items():
            logger.info(f"Đang xử lý file: {file_name}")
            
            base_metadata = {
                'file_name': file_name,
                'processed_date': datetime.now().isoformat()
            }
            
            # Tạo chunks cho file hiện tại
            file_chunks = processor.process_text(content, base_metadata)
            
            chunk_objects = [
                TextChunk(
                    content=chunk['content'],
                    metadata=chunk['metadata']
                ) for chunk in file_chunks
            ]
            
            # Lưu chunks của file hiện tại
            try:
                output_file = save_chunks_to_json(chunk_objects, OUTPUT_DIR, file_name)
                logger.info(f"Đã xử lý xong file {file_name}")
            except Exception as e:
                logger.error(f"Lỗi khi lưu kết quả cho file {file_name}: {str(e)}")
                continue
    
    except Exception as e:
        logger.error(f"Lỗi trong quá trình xử lý: {str(e)}")

if __name__ == "__main__":
    main()
