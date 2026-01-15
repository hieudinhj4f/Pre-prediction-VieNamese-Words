import os
import collections
import pickle
import re
from underthesea import word_tokenize
from underthesea.dictionary import Dictionary

class VietnamesePredictor:
    #Count trigram, bigram, unigram
    def __init__(self):
        self.unigram_counts = collections.Counter()
        self.bigram_counts = collections.Counter()
        self.trigram_counts = collections.Counter()
    #Chuẩn hóa văn bản
    def clean_text(self, text):
        # Sử dụng Lowercase và loại bỏ khoảng trắng thừa
        text = text.lower().strip()
        # Sử dụng Regex để loại bỏ ký tự đặc biệt, chỉ giữ lại chữ cái và số
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def train_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                text = self.clean_text(line)
                if not text: continue
                # Tokenize tiếng Việt
                tokens = word_tokenize(text, format="text").split()
                # Cập nhật các bộ đếm( Cập nhật trigram, bigram, unigram)
                self.unigram_counts.update(tokens)
                if len(tokens) >= 2:
                    self.bigram_counts.update(zip(tokens, tokens[1:]))
                if len(tokens) >= 3:
                    self.trigram_counts.update(zip(tokens, tokens[1:], tokens[2:]))
    #Bổ sung thêm từ điển nội bộ
    def inject_internal_dictionary(self):
        print(" Đang nạp từ điển nội bộ từ Underthesea ")
        try:
            # Lấy từ điển singleton của Underthesea
            dic = Dictionary.instance()
            words = dic.words
            for w in words:
                word = w.lower()
                if word not in self.unigram_counts:
                    self.unigram_counts[word] = 1
        except Exception as e:
            print(f" Không thể nạp từ điển nội bộ: {e}")
            
    # Lưu mô hình đã huấn luyện
    def save_model(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f" Đã lưu mô hình tại: {path}")
    # Nạp mô hình đã lưu
    @staticmethod
    def load_model(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    # Dự đoán từ tiếp theo dựa trên ngữ cảnh và tiền tố
    def predict(self, context_tuple, prefix, top_n=5):
        w1, w2 = [w.lower() for w in context_tuple]
        pre = prefix.lower()
        scores = {}

        # Lấy kích thước từ vựng (V) để làm trơn
        V = len(self.unigram_counts)
        if V == 0: V = 1 # Tránh chia cho 0 nếu chưa train


        total_tri = sum([count for (tw1, tw2, tw3), count in self.trigram_counts.items() if tw1 == w1 and tw2 == w2])
        #3.Trigram với Laplace
        for (tw1, tw2, tw3), count in self.trigram_counts.items():
            if tw1 == w1 and tw2 == w2 and tw3.startswith(pre):
                # Công thức Laplace: (count + 1) / (tổng ngữ cảnh + V)
                prob = (count + 1) / (total_tri + V)
                scores[tw3] = scores.get(tw3, 0) + prob * 10

        # 2. Bigram Score với Laplace
        total_bi = sum([count for (bw1, bw2), count in self.bigram_counts.items() if bw1 == w2])
        
        for (bw1, bw2), count in self.bigram_counts.items():
            if bw1 == w2 and bw2.startswith(pre):
                prob = (count + 1) / (total_bi + V)
                scores[bw2] = scores.get(bw2, 0) + prob * 1.0

        # 3. Unigram Score với Laplace
        total_uni = sum(self.unigram_counts.values())
        
        for word, count in self.unigram_counts.items():
            if word.startswith(pre):
                prob = (count + 1) / (total_uni + V)
                if word not in scores: # Ưu tiên Tri và Bi đã tính trước
                    scores[word] = scores.get(word, 0) + prob * 0.1

        # Lấy top N kết quả
        results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [word.replace("_", " ") for word, _ in results[:top_n]]
    

    def update_learning(self, sentence):
        # Ứng dụng giúp mô  hình học thêm từ câu người dùng vừa nhập
        text = self.clean_text(sentence)
        if not text: return
        
        tokens = word_tokenize(text, format="text").split()
        
        # Cập nhật các bộ đếm ngay lập tức
        self.unigram_counts.update(tokens)
        if len(tokens) >= 2:
            self.bigram_counts.update(zip(tokens, tokens[1:]))
        if len(tokens) >= 3:
            self.trigram_counts.update(zip(tokens, tokens[1:], tokens[2:]))
        