import os
import collections
import pickle
import re
from underthesea import word_tokenize
from underthesea.dictionary import Dictionary

class VietnamesePredictor:
    def __init__(self):
        self.unigram_counts = collections.Counter()
        self.bigram_counts = collections.Counter()
        self.trigram_counts = collections.Counter()

    def clean_text(self, text):
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def train_file(self, file_path):
        """Đọc và học từ tệp văn bản"""
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                text = self.clean_text(line)
                if not text: continue
                # Tokenize tiếng Việt
                tokens = word_tokenize(text, format="text").split()
                
                self.unigram_counts.update(tokens)
                if len(tokens) >= 2:
                    self.bigram_counts.update(zip(tokens, tokens[1:]))
                if len(tokens) >= 3:
                    self.trigram_counts.update(zip(tokens, tokens[1:], tokens[2:]))

    def inject_internal_dictionary(self):
        """Nạp từ điển nội bộ của underthesea (Sửa lỗi Singleton)"""
        print("--- 📚 Đang nạp từ điển nội bộ từ Underthesea ---")
        try:
            # Cách gọi đúng cho Singleton trong underthesea
            dic = Dictionary.instance()
            words = dic.words
            for w in words:
                word = w.lower()
                if word not in self.unigram_counts:
                    self.unigram_counts[word] = 1
        except Exception as e:
            print(f"⚠️ Không thể nạp từ điển nội bộ: {e}")

    def save_model(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"✅ Đã lưu mô hình tại: {path}")

    @staticmethod
    def load_model(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def predict(self, context_tuple, prefix, top_n=5):
        w1, w2 = [w.lower() for w in context_tuple]
        pre = prefix.lower()
        scores = collections.defaultdict(float)

        # Trigram (Trọng số 10)
        for (tw1, tw2, tw3), count in self.trigram_counts.items():
            if tw1 == w1 and tw2 == w2 and tw3.startswith(pre):
                scores[tw3] += count * 10
        # Bigram (Trọng số 1)
        if len(scores) < top_n:
            for (bw1, bw2), count in self.bigram_counts.items():
                if bw1 == w2 and bw2.startswith(pre):
                    scores[bw2] += count
        # Unigram (Trọng số 0.01)
        if len(scores) < top_n:
            for word, count in self.unigram_counts.items():
                if word.startswith(pre) and word not in scores:
                    scores[word] += count * 0.01

        results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [word.replace("_", " ") for word, _ in results[:top_n]]
    

    def update_learning(self, sentence):
        """Hàm giúp mô hình học thêm từ câu người dùng vừa nhập"""
        text = self.clean_text(sentence)
        if not text: return
        
        tokens = word_tokenize(text, format="text").split()
        
        # Cập nhật các bộ đếm ngay lập tức
        self.unigram_counts.update(tokens)
        if len(tokens) >= 2:
            self.bigram_counts.update(zip(tokens, tokens[1:]))
        if len(tokens) >= 3:
            self.trigram_counts.update(zip(tokens, tokens[1:], tokens[2:]))
        
        # Lưu lại mô hình ngay để "ghi nhớ" vĩnh viễn
        # (Lưu ý: Trong thực tế nếu dữ liệu lớn thì nên lưu định kỳ để tránh chậm máy)
        # self.save_model("vietnamese_ngram_mega.pkl")