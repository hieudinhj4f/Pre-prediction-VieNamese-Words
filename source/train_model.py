import os
from predictor import VietnamesePredictor

# Lấy thư mục hiện tại của file train_model.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Cấu hình đường dẫn linh hoạt
CORPUS_FILES = [
    os.path.join(BASE_DIR, "news-corpus", "sample", "demo-full.txt"),
    os.path.join(BASE_DIR, "news-corpus", "sample", "demo-title.txt"),
    os.path.join(BASE_DIR, "vietnamese-wordlist", "Viet11K.txt"),
    os.path.join(BASE_DIR, "vietnamese-wordlist", "Viet22K.txt"),
    os.path.join(BASE_DIR, "vietnamese-wordlist", "Viet39K.txt"),
    os.path.join(BASE_DIR, "vietnamese-wordlist", "Viet74K.txt")
]

MODEL_PATH = os.path.join(BASE_DIR, "vietnamese_ngram_mega.pkl")

# Các phần còn lại của hàm main() giữ nguyên...
def main():
    predictor = VietnamesePredictor()
    
    print("--- 🔄 BẮT ĐẦU QUY TRÌNH HỢP NHẤT 6 BỘ DỮ LIỆU ---")
    
    # Gộp TRƯỚC khi lưu: Học từ 6 file văn bản
    for path in CORPUS_FILES:
        if os.path.exists(path):
            print(f"👉 Đang học ngữ cảnh từ: {os.path.basename(path)}")
            predictor.train_file(path)
        else:
            print(f"⚠️ Bỏ qua (không tìm thấy): {path}")

    # Bổ sung SAU khi học xong văn bản: Nạp từ điển nội bộ
    predictor.inject_internal_dictionary()
    
    # Xuất ra file "tinh hoa" duy nhất
    predictor.save_model(MODEL_PATH)
    print(f"\n✅ HOÀN TẤT! Mô hình Mega chứa tri thức từ 6 file đã sẵn sàng.")

if __name__ == "__main__":
    main()