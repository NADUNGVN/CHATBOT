import PyPDF2
import os

def pdf_to_text(pdf_path, output_dir):
    try:
        # Đảm bảo thư mục output tồn tại
        os.makedirs(output_dir, exist_ok=True)
        
        # Lấy tên file từ đường dẫn PDF
        pdf_filename = os.path.basename(pdf_path)
        output_filename = os.path.splitext(pdf_filename)[0] + '.txt'
        output_path = os.path.join(output_dir, output_filename)
        
        # Mở file PDF
        with open(pdf_path, 'rb') as file:
            # Tạo đối tượng PDF reader
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Tạo một string để lưu text
            text = ""
            
            # Đọc từng trang và thêm vào text
            for page in pdf_reader.pages:
                text += page.extract_text()
            
            # Ghi text vào file output
            with open(output_path, 'w', encoding='utf-8') as output_file:
                output_file.write(text)
                
            print(f"Chuyển đổi thành công! File text được lưu tại: {output_path}")
            
    except Exception as e:
        print(f"Có lỗi xảy ra: {str(e)}")

# Đường dẫn đã được cung cấp
pdf_path = r"C:\Users\nguye\OneDrive\Máy tính\CHATBOT_pdf_chunk_4_2\experimental_1\data\pdf\1.000656.000.00.00.H29.pdf"
output_dir = r"C:\Users\nguye\OneDrive\Máy tính\CHATBOT_pdf_chunk_4_2\experimental_1\data\text"

# Thực hiện chuyển đổi
pdf_to_text(pdf_path, output_dir)
