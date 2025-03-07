import fitz  # PyMuPDF
import os

def pdf_to_text(pdf_path, output_dir):
    """
    Trích xuất văn bản từ PDF và lưu dưới dạng file .txt để kiểm tra.
    """
    # Đảm bảo thư mục đầu ra tồn tại
    os.makedirs(output_dir, exist_ok=True)

    # Lấy tên file từ đường dẫn PDF
    pdf_filename = os.path.basename(pdf_path)
    output_filename = os.path.splitext(pdf_filename)[0] + '.txt'
    output_path = os.path.join(output_dir, output_filename)

    doc = fitz.open(pdf_path)
    text = ""

    for page in doc:
        text += page.get_text("text") + "\n\n"  # Xuống dòng giữa các trang

    # Ghi text vào file output
    with open(output_path, 'w', encoding='utf-8') as output_file:
        output_file.write(text)
    
    print(f"Chuyển đổi thành công! File text được lưu tại: {output_path}")

# Chạy với đường dẫn PDF cụ thể của bạn
pdf_path = r"C:\Users\nguye\OneDrive\Máy tính\CHATBOT_pdf_chunk_4_2\experimental_2\data\pdf\1.000656.000.00.00.H29.pdf"
output_dir = r"C:\Users\nguye\OneDrive\Máy tính\CHATBOT_pdf_chunk_4_2\experimental_2\data\text"

pdf_to_text(pdf_path, output_dir)

# pdf_path = r"C:\Users\nguye\OneDrive\Máy tính\CHATBOT_pdf_chunk_4_2\experimental_1\data\pdf\1.000656.000.00.00.H29.pdf"
# output_dir = r"C:\Users\nguye\OneDrive\Máy tính\CHATBOT_pdf_chunk_4_2\experimental_1\data\text"
