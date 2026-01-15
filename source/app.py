import streamlit as st
from predictor import VietnamesePredictor
import time

# --- CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="Vietnamese Word Predictor",
    page_icon="⌨️",
    layout="centered"
)

# --- LOAD MODEL 
import os

@st.cache_resource
def load_model():
    # 1. Danh sách các đường dẫn có khả năng chứa file mô hình
    # Thử tìm ở thư mục hiện tại, sau đó thử tìm trong thư mục 'source/'
    possible_paths = [
        "vietnamese_ngram_mega.pkl", 
        "source/vietnamese_ngram_mega.pkl",
        os.path.join(os.path.dirname(__file__), "vietnamese_ngram_mega.pkl")
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                return VietnamesePredictor.load_model(path)
            except Exception as e:
                st.error(f"Lỗi khi mở file {path}: {e}")
    
    # 2. Nếu không tìm thấy ở bất cứ đâu
    st.error("KHÔNG TÌM THẤY FILE MÔ HÌNH!")
    st.info("Hãy đảm bảo file 'vietnamese_ngram_mega.pkl' đã được upload lên GitHub cùng thư mục với app.py")
    return None

predictor = load_model()

# --- GIAO DIỆN CHÍNH ---
st.title("Bộ gõ Tiếng Việt Thông minh")
st.markdown("""
Ứng dụng sử dụng mô hình **N-Gram (Trigram)** kết hợp dữ liệu từ 6 nguồn văn bản và từ điển nội bộ để dự báo từ tiếp theo.
""")

# --- PHẦN 1: NHẬP LIỆU & DỰ BÁO ---
st.subheader("📝 Soạn thảo văn bản")

# Tạo 2 cột: 1 cho văn bản chính, 1 cho ký tự đang gõ
col1, col2 = st.columns([3, 1])

with col1:
    input_text = st.text_input("Văn bản đã gõ:", placeholder="Ví dụ: tôi đang", key="main_input")

with col2:
    prefix = st.text_input("Từ đang gõ dở:", placeholder="h", help="Ký tự đầu của từ tiếp theo")

# Xử lý dự báo
if input_text:
    words = input_text.strip().split()
    if len(words) >= 2:
        context = (words[-2], words[-1])
        # Gọi hàm predict từ predictor.py
        suggestions = predictor.predict(context, prefix)
        
        if suggestions:
            st.write("**Gợi ý từ tiếp theo:**")
            # Hiển thị gợi ý dạng nút bấm ngang
            cols = st.columns(len(suggestions))
            for i, word in enumerate(suggestions):
                if cols[i].button(word, use_container_width=True):
                    st.info(f"Bạn đã chọn: **{word}**")
                    st.balloons() # Hiệu ứng chúc mừng khi chọn từ
        else:
            st.caption(" Không tìm thấy gợi ý phù hợp. Hãy thử gõ thêm ký tự dở.")
    else:
        st.warning("Hãy nhập ít nhất 2 từ để bắt đầu dự báo.")

st.divider()

# --- PHẦN 2: TÍNH NĂNG TỰ HỌC (Ghi điểm BTL) ---
st.subheader("🧠 Giúp máy thông minh hơn")
new_sentence = st.text_area("Nhập một câu mới để dạy máy (ví dụ: Xin chào bạn):")

if st.button("Dạy máy câu này"):
    if new_sentence:
        with st.spinner('Đang học...'):
            # Gọi hàm update_learning (bạn đã thêm vào predictor.py)
            if hasattr(predictor, 'update_learning'):
                predictor.update_learning(new_sentence)
                predictor.save_model("vietnamese_ngram_mega.pkl")
                time.sleep(1)
                st.success("Tuyệt vời! Máy đã ghi nhớ câu này và sẽ gợi ý tốt hơn lần sau.")
            else:
                st.error("Lỗi: Hàm update_learning chưa được thêm vào predictor.py")
    else:
        st.error("Vui lòng nhập văn bản trước khi nhấn dạy máy.")

# --- SIDEBAR (Hướng dẫn) ---
st.sidebar.header("Thông tin dự án")
st.sidebar.info("""
- **Mô hình:** Hybrid N-Gram
- **Dữ liệu:** 6 nguồn văn bản + Dictionary
- **Tính năng:** Dự báo thời gian thực & Học tăng cường (Online Learning)
""")