import re
import json

def text_to_json(text_file_path, output_json_path):
    # Đọc nội dung file text
    with open(text_file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Khởi tạo dictionary để lưu thông tin
    thu_tuc = {}

    def extract(pattern, content, default=""):
        match = re.search(pattern, content, re.DOTALL)
        return match.group(1).strip() if match else default

    def clean_text(text):
        if not text:
            return ""
        # Loại bỏ khoảng trắng thừa và ký tự đặc biệt không cần thiết
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text

    # Lấy thông tin cơ bản
    thu_tuc['ma_thu_tuc'] = extract(r'Mã thủ tục:\s*(.*?)\s*\n', content)
    thu_tuc['so_quyet_dinh'] = extract(r'Số quyết định:\s*(.*?)\s*\n', content)
    thu_tuc['ten_thu_tuc'] = extract(r'Tên thủ tục:\s*(.*?)\s*\n', content)
    thu_tuc['cap_thuc_hien'] = extract(r'Cấp thực hiện:\s*(.*?)\s*\n', content)
    thu_tuc['loai_thu_tuc'] = extract(r'Loại thủ tục:\s*(.*?)\s*\n', content)
    thu_tuc['linh_vuc'] = extract(r'Lĩnh vực:\s*(.*?)\s*\n', content)

    # Trích xuất trình tự thực hiện
    trinh_tu_text = extract(r'Trình tự thực hiện:\s*\n(.*?)Cách thức thực hiện:', content)
    if trinh_tu_text:
        luu_y = extract(r'Lưu ý:\s*\n(.*?)Bước 1:', trinh_tu_text)
        buoc_pattern = r'Bước (\d+):(.*?)(?=Bước \d+:|$)'
        buoc_matches = re.findall(buoc_pattern, trinh_tu_text, re.DOTALL)
        
        thu_tuc['trinh_tu_thuc_hien'] = {
            'luu_y': clean_text(luu_y),
            'cac_buoc': [{'buoc': int(buoc[0]), 'noi_dung': clean_text(buoc[1])} for buoc in buoc_matches]
        }

    # Trích xuất cách thức thực hiện
    cach_thuc_text = extract(r'Cách thức thực hiện:\s*\n(.*?)Thành phần hồ sơ:', content)
    if cach_thuc_text:
        cach_thuc = []
        
        # Xử lý cho từng hình thức
        hinh_thuc_patterns = [
            ('Trực tiếp', r'Trực\s+tiếp\s*\n(.*?)(?=Trực\s+tuyến|Dịch\s+vụ\s+bưu\s+chính|$)'),
            ('Trực tuyến', r'Trực\s+tuyến\s*\n(.*?)(?=Dịch\s+vụ\s+bưu\s+chính|$)'),
            ('Dịch vụ bưu chính', r'Dịch\s+vụ\s+bưu\s+chính\s*\n(.*?)(?=\n\n|$)')
        ]
        
        for hinh_thuc_name, pattern in hinh_thuc_patterns:
            match = re.search(pattern, cach_thuc_text, re.DOTALL)
            if match:
                noi_dung = match.group(1)
                
                # Trích xuất thông tin chi tiết
                thoi_han = extract(r'(?:Ngay|thời hạn giải quyết).*?:(.*?)(?=Phí|$)', noi_dung, re.IGNORECASE)
                phi_le_phi = extract(r'(?:Phí|Lệ phí).*?:(.*?)(?=\.|$)', noi_dung, re.DOTALL)
                mo_ta = extract(r'(?:Mô tả|:)\s*(.*?)(?=\n\n|$)', noi_dung)
                
                cach_thuc.append({
                    'hinh_thuc': hinh_thuc_name,
                    'chi_tiet': {
                        'thoi_han_giai_quyet': clean_text(thoi_han),
                        'phi_le_phi': clean_text(phi_le_phi),
                        'mo_ta': clean_text(mo_ta)
                    }
                })
        
        thu_tuc['cach_thuc_thuc_hien'] = cach_thuc

    # Trích xuất thành phần hồ sơ
    ho_so_match = re.search(r'Thành phần hồ sơ:\s*\n(.*?)Đối tượng thực hiện:', content, re.DOTALL)
    if ho_so_match:
        ho_so_text = ho_so_match.group(1).strip()
        
        # Phân tách giấy tờ phải nộp và xuất trình
        giay_to_nop_match = re.search(r'\* Giấy tờ phải nộp:(.*?)\* Giấy tờ phải xuất trình:', ho_so_text, re.DOTALL)
        giay_to_xuat_trinh_match = re.search(r'\* Giấy tờ phải xuất trình:(.*?)\* Lưu ý :', ho_so_text, re.DOTALL)
        luu_y_ho_so_match = re.search(r'\* Lưu ý :(.*?)Bao gồm', ho_so_text, re.DOTALL)
        bao_gom_match = re.search(r'Bao gồm\s*(.*?)Đối tượng thực hiện:', ho_so_text, re.DOTALL)
        
        ho_so = {}
        
        if giay_to_nop_match:
            giay_to_nop_text = giay_to_nop_match.group(1).strip()
            giay_to_nop_pattern = r'- (.*?)(?=\n\s*-|\n\s*\*|$)'
            giay_to_nop = re.findall(giay_to_nop_pattern, giay_to_nop_text, re.DOTALL)
            ho_so['giay_to_phai_nop'] = [gt.strip() for gt in giay_to_nop if gt.strip()]
        
        if giay_to_xuat_trinh_match:
            giay_to_xuat_trinh_text = giay_to_xuat_trinh_match.group(1).strip()
            giay_to_xuat_trinh_pattern = r'- (.*?)(?=\n\s*-|\n\s*\*|$)'
            giay_to_xuat_trinh = re.findall(giay_to_xuat_trinh_pattern, giay_to_xuat_trinh_text, re.DOTALL)
            ho_so['giay_to_phai_xuat_trinh'] = [gt.strip() for gt in giay_to_xuat_trinh if gt.strip()]
        
        if luu_y_ho_so_match:
            luu_y_ho_so_text = luu_y_ho_so_match.group(1).strip()
            luu_y_ho_so_pattern = r'- (.*?)(?=\n\s*-|\n\s*\*|$)'
            luu_y_ho_so = re.findall(luu_y_ho_so_pattern, luu_y_ho_so_text, re.DOTALL)
            ho_so['luu_y'] = [ly.strip() for ly in luu_y_ho_so if ly.strip()]
        
        if bao_gom_match:
            bao_gom_text = bao_gom_match.group(1).strip()
            bao_gom_pattern = r'- (.*?)(?=\n\s*-|\n\s*\*|$)'
            bao_gom = re.findall(bao_gom_pattern, bao_gom_text, re.DOTALL)
            ho_so['bao_gom'] = [bg.strip() for bg in bao_gom if bg.strip()]
        
        thu_tuc['thanh_phan_ho_so'] = ho_so

    # Trích xuất các thông tin khác
    thu_tuc['doi_tuong_thuc_hien'] = extract(r'Đối tượng thực hiện:\s*(.*?)\s*\n', content)
    thu_tuc['co_quan_thuc_hien'] = extract(r'Cơ quan thực hiện:\s*(.*?)\s*\n', content)
    thu_tuc['co_quan_tham_quyen'] = extract(r'Cơ quan có thẩm quyền:\s*(.*?)\s*\n', content)
    thu_tuc['dia_chi_tiep_nhan'] = extract(r'Địa chỉ tiếp nhận HS:\s*(.*?)\s*\n', content)
    thu_tuc['ket_qua_thuc_hien'] = extract(r'Kết quả thực hiện:\s*(.*?)\s*\n', content)

    # Trích xuất căn cứ pháp lý
    can_cu_text = extract(r'Căn cứ pháp lý:\s*\n(.*?)Yêu cầu, điều kiện thực hiện:', content)
    if can_cu_text:
        can_cu_pattern = r'(\S+)\s*\n\s*(.*?)\s*\n\s*(\d{2}-\d{2}-\d{4})\s*\n\s*(.*?)\s*\n'
        can_cu_matches = re.findall(can_cu_pattern, can_cu_text)

        thu_tuc['can_cu_phap_ly'] = [
            {
                'so_ky_hieu': clean_text(cc[0]),
                'trich_yeu': clean_text(cc[1]),
                'ngay_ban_hanh': clean_text(cc[2]),
                'co_quan_ban_hanh': clean_text(cc[3])
            } 
            for cc in can_cu_matches
        ]

    # Trích xuất yêu cầu điều kiện thực hiện
    thu_tuc['yeu_cau_dieu_kien'] = extract(r'Yêu cầu, điều kiện thực hiện:\s*(.*?)\s*\n', content)

    # Ghi kết quả ra file JSON
    with open(output_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(thu_tuc, json_file, ensure_ascii=False, indent=2)

    return thu_tuc

if __name__ == "__main__":
    input_dir = "C:/Users/nguye/OneDrive/Máy tính/CHATBOT_pdf_chunk_4_2/experimental_2/data/text/1.000656.000.00.00.H29.txt"
    output_dir = "C:/Users/nguye/OneDrive/Máy tính/CHATBOT_pdf_chunk_4_2/experimental_2/data/json"
    
    result = text_to_json(input_dir, output_dir)
    print(f"Đã chuyển đổi thành công và lưu kết quả tại: {output_dir}")
