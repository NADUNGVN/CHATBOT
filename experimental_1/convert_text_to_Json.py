import re
import json
import os

def text_to_json(text_file_path, output_json_path):
    # Đọc nội dung file text
    with open(text_file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Khởi tạo dictionary để lưu thông tin
    thu_tuc = {}
    
    # Lấy thông tin cơ bản
    thu_tuc['ma_thu_tuc'] = re.search(r'Mã thủ tục:\s*(.*?)\s*\n', content).group(1).strip()
    thu_tuc['so_quyet_dinh'] = re.search(r'Số quyết định:\s*(.*?)\s*\n', content).group(1).strip()
    thu_tuc['ten_thu_tuc'] = re.search(r'Tên thủ tục:\s*(.*?)\s*\n', content).group(1).strip()
    thu_tuc['cap_thuc_hien'] = re.search(r'Cấp thực hiện:\s*(.*?)\s*\n', content).group(1).strip()
    thu_tuc['loai_thu_tuc'] = re.search(r'Loại thủ tục:\s*(.*?)\s*\n', content).group(1).strip()
    thu_tuc['linh_vuc'] = re.search(r'Lĩnh vực:\s*(.*?)\s*\n', content).group(1).strip()
    
    # Trích xuất trình tự thực hiện
    trinh_tu_match = re.search(r'Trình tự thực hiện:\s*\n(.*?)Cách thức thực hiện:', content, re.DOTALL)
    if trinh_tu_match:
        trinh_tu_text = trinh_tu_match.group(1).strip()
        
        # Tách phần lưu ý và các bước
        luu_y_match = re.search(r'Lưu ý:\s*\n(.*?)Bước 1:', trinh_tu_text, re.DOTALL)
        luu_y = luu_y_match.group(1).strip() if luu_y_match else ""
        
        # Tìm tất cả các bước
        buoc_pattern = r'Bước (\d+):(.*?)(?=Bước \d+:|$)'
        buoc_matches = re.findall(buoc_pattern, trinh_tu_text, re.DOTALL)
        
        trinh_tu = {
            'luu_y': luu_y,
            'cac_buoc': [{'buoc': int(buoc[0]), 'noi_dung': buoc[1].strip()} for buoc in buoc_matches]
        }
        
        thu_tuc['trinh_tu_thuc_hien'] = trinh_tu
    
    # Trích xuất cách thức thực hiện
    cach_thuc_match = re.search(r'Cách thức thực hiện:\s*\n(.*?)Thành phần hồ sơ:', content, re.DOTALL)
    if cach_thuc_match:
        cach_thuc_text = cach_thuc_match.group(1).strip()
        
        # Tìm các hình thức nộp
        hinh_thuc_pattern = r'(Trực tiếp|Trực tuyến|Dịch vụ bưu chính)\s*\n\s*(.*?)(?=Trực tiếp|Trực tuyến|Dịch vụ bưu chính|\n\nThành phần hồ sơ)'
        hinh_thuc_matches = re.findall(hinh_thuc_pattern, cach_thuc_text, re.DOTALL)
        
        cach_thuc = []
        for hinh_thuc in hinh_thuc_matches:
            ten_hinh_thuc = hinh_thuc[0].strip()
            chi_tiet = hinh_thuc[1].strip()
            
            # Tách thời hạn và phí lệ phí
            thoi_han_match = re.search(r'(Ngay trong ngày.*?)Phí', chi_tiet, re.DOTALL)
            thoi_han = thoi_han_match.group(1).strip() if thoi_han_match else ""
            
            phi_le_phi_pattern = r'Phí\s*:\s*(.*?)(?=Lệ phí|$)'
            le_phi_pattern = r'Lệ phí\s*:\s*(.*?)(?=Phí|$)'
            
            phi_matches = re.findall(phi_le_phi_pattern, chi_tiet)
            le_phi_matches = re.findall(le_phi_pattern, chi_tiet)
            
            phi = [p.strip() for p in phi_matches if p.strip()]
            le_phi = [lp.strip() for lp in le_phi_matches if lp.strip()]
            
            cach_thuc.append({
                'hinh_thuc': ten_hinh_thuc,
                'thoi_han': thoi_han,
                'phi': phi,
                'le_phi': le_phi
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
    
    # Trích xuất thông tin khác
    thu_tuc['doi_tuong_thuc_hien'] = re.search(r'Đối tượng thực hiện:\s*(.*?)\s*\n', content).group(1).strip()
    thu_tuc['co_quan_thuc_hien'] = re.search(r'Cơ quan thực hiện:\s*(.*?)\s*\n', content).group(1).strip()
    thu_tuc['co_quan_tham_quyen'] = re.search(r'Cơ quan có thẩm quyền:\s*(.*?)\s*\n', content).group(1).strip()
    thu_tuc['dia_chi_tiep_nhan'] = re.search(r'Địa chỉ tiếp nhận HS:\s*(.*?)\s*\n', content).group(1).strip()
    
    # Trích xuất kết quả thực hiện
    ket_qua_match = re.search(r'Kết quả thực hiện:\s*(.*?)\s*\n', content)
    if ket_qua_match:
        thu_tuc['ket_qua_thuc_hien'] = ket_qua_match.group(1).strip()
    
    # Trích xuất căn cứ pháp lý
    can_cu_match = re.search(r'Căn cứ pháp lý:\s*\n(.*?)Yêu cầu, điều kiện thực hiện:', content, re.DOTALL)
    if can_cu_match:
        can_cu_text = can_cu_match.group(1).strip()
        can_cu_pattern = r'(\S+)\s*\n\s*(.*?)\s*\n\s*(\d{2}-\d{2}-\d{4})\s*\n\s*(.*?)\s*\n'
        can_cu_matches = re.findall(can_cu_pattern, can_cu_text)
        
        can_cu = []
        for cc in can_cu_matches:
            can_cu.append({
                'so_ky_hieu': cc[0].strip(),
                'trich_yeu': cc[1].strip(),
                'ngay_ban_hanh': cc[2].strip(),
                'co_quan_ban_hanh': cc[3].strip()
            })
        
        thu_tuc['can_cu_phap_ly'] = can_cu
    
    # Trích xuất yêu cầu, điều kiện thực hiện
    yeu_cau_match = re.search(r'Yêu cầu, điều kiện thực hiện:\s*(.*?)\s*\n', content)
    if yeu_cau_match:
        thu_tuc['yeu_cau_dieu_kien'] = yeu_cau_match.group(1).strip()
    
    # Ghi kết quả ra file JSON
    with open(output_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(thu_tuc, json_file, ensure_ascii=False, indent=2)
    
    return thu_tuc

if __name__ == "__main__":
    input_dir = "C:/Users/nguye/OneDrive/Máy tính/CHATBOT_pdf_chunk_4_2/experimental_1/data/text/1.000656.000.00.00.H29.txt"
    output_dir = "C:/Users/nguye/OneDrive/Máy tính/CHATBOT_pdf_chunk_4_2/experimental_1/data/json"
    
    result = text_to_json(input_dir, output_dir)
    print(f"Đã chuyển đổi thành công và lưu kết quả tại: {output_dir}")
