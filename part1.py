import numpy as np
from collections import defaultdict

class word_vectors():
    def __init__(self):
        self.UNK = '<UNK>'
        self.raw_count = None
        self.ppmi = None
        self.reduced_ppmi = None
        self.word_dict = defaultdict(int) # dict from word -> index
        self.reversed_word_dict = defaultdict(str) # dict from index -> word

    def train(self, train_data):
        # Build word_dict
        with open(train_data) as f:
            lines = f.readlines()
            for line in lines:
                words = line.lower().split()
                for word in words:
                    if word in self.word_dict:
                        continue
                    next_index = len(self.word_dict)
                    self.word_dict[word] = next_index
                    self.reversed_word_dict[next_index] = word
        unk_index = len(self.word_dict)
        self.word_dict[self.UNK] = unk_index
        self.reversed_word_dict[unk_index] = self.UNK

        # Initialize raw_count
        vocab_size = len(self.word_dict)
        self.raw_count = np.zeros((vocab_size, vocab_size))

        # Build raw_count
        with open(train_data) as f:
            lines = f.readlines()
            for line in lines:
                words = line.lower().split()
                for i in range(len(words)):
                    cur_word = words[i]
                    cur_word_index = self.word_dict.get(cur_word, self.word_dict[self.UNK])
                    # If there is a previous word
                    if i > 0:
                        prev_word = words[i - 1]
                        prev_word_index = self.word_dict.get(prev_word, self.word_dict[self.UNK])
                        self.raw_count[cur_word_index][prev_word_index] += 1
                    if i < len(words) - 1:
                        next_word = words[i + 1]
                        next_word_index = self.word_dict.get(next_word, self.word_dict[self.UNK])
                        self.raw_count[cur_word_index][next_word_index] += 1
        self.raw_count = np.multiply(self.raw_count, 10) 
        self.raw_count = np.add(self.raw_count, 1)
        self.raw_count = np.multiply(self.raw_count, 1 / self.raw_count.sum())
        
        # Compute PPMI
        pw = self.raw_count.sum(axis = 1)
        pc = self.raw_count.sum(axis = 0)
        self.ppmi = self.raw_count.copy()
        self.ppmi /= pc
        self.ppmi /= pw[:, np.newaxis]
        # self.ppmi /= pw
        f = lambda x: 0.0 if x <= 1 else np.log(x) 
        f = np.vectorize(f)
        self.ppmi = f(self.ppmi)
    
    def get_word_vec_ppmi(self, word):
        word_index = self.word_dict.get(word, self.word_dict[self.UNK])
        return self.ppmi[word_index].copy()
    
    def get_word_vec_raw(self, word):
        word_index = self.word_dict.get(word, self.word_dict[self.UNK])
        return self.raw_count[word_index].copy()
    
    def compute_distance(self, word1, word2):
        vec1 = self.get_word_vec_ppmi(word1)
        vec2 = self.get_word_vec_ppmi(word2)
        diff = vec1 - vec2
        return np.linalg.norm(diff)


if __name__ == '__main__':
    wvs = word_vectors()
    wvs.train('data/dist_sim_data.txt')
    print(wvs.get_word_vec_raw('dogs'))
    print(wvs.get_word_vec_ppmi('dogs'))
    print(wvs.compute_distance("women", "men"))
    print(wvs.compute_distance("women", "dogs"))
    print(wvs.compute_distance("men", "dogs"))
    print(wvs.compute_distance("feed", "like"))
    print(wvs.compute_distance("feed", "bite"))
    print(wvs.compute_distance("like", "bite"))

