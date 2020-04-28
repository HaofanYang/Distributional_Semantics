import numpy as np
from collections import defaultdict
import scipy.linalg as scipy_linalg

class word_vectors():
    USE_PPMI = 'PPMI'
    USE_REDUCED_PPMI = 'REDUCED_PPMI'
    USE_RAW_COUNT = 'RAW_COUNT'

    def __init__(self):
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
                    cur_word_index = self.word_dict[cur_word]
                    # If there is a previous word
                    if i > 0:
                        prev_word = words[i - 1]
                        prev_word_index = self.word_dict[prev_word]
                        self.raw_count[cur_word_index][prev_word_index] += 1
                    if i < len(words) - 1:
                        next_word = words[i + 1]
                        next_word_index = self.word_dict[next_word]
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

        # Build reduced PPMI
        U, E, Vt = scipy_linalg.svd(self.ppmi, full_matrices=False)
        U = np.matrix(U) # compute U
        E = np.matrix(np.diag(E)) # compute E
        Vt = np.matrix(Vt) # compute Vt = conjugage transpose of V
        V = Vt.T # compute V = conjugate transpose of Vt
        UE = np.matrix(np.matmul(U, E))
        UEVt = np.matrix(np.matmul(UE, Vt))
        self.__verify_SVD_reconstruction(UEVt)
        self.reduced_ppmi = self.ppmi * V[:, 0:3]

    def get_word_vec(self, word, option, print_info = False):
        switcher = {
            word_vectors.USE_PPMI: self.ppmi,
            word_vectors.USE_REDUCED_PPMI: self.reduced_ppmi,
            word_vectors.USE_RAW_COUNT: self.raw_count,
        }
        word_index = self.word_dict[word]
        matrix = switcher[option]
        word_vector = matrix[word_index]
        if print_info:
            print("The word vector for \'{w}\', using {o}: \n{vec}".format(w = word, o = option, vec = word_vector))
        return word_vector
    
    def compute_distance(self, word1, word2, option, print_info = False):
        vec1 = self.get_word_vec(word1, option)
        vec2 = self.get_word_vec(word2, option)
        diff = vec1 - vec2
        distance = np.linalg.norm(diff)
        if print_info:
            print("The difference between \'{w1}\' and \'{w2}\' using [{op}] is {d}".format(w1 = word1, w2 = word2, op = option, d = distance))
        return distance
    
    # Verify that matrices multiplications reproduce the original PPMI matrix
    # Print error messages if failed to do so
    def __verify_SVD_reconstruction(self, UEVt):
        diff = self.ppmi - UEVt
        filter_small = lambda x : False if abs(x) < 1e-14 else True
        filter_small = np.vectorize(filter_small)
        booleans = filter_small(diff)
        large_elements = np.extract(booleans, diff)
        if len(large_elements) != 0:
            print("===============Failed to reconstruct PPMI matrix from SVD===============")
            print("===============DIFF (PPMI - UEVt)===============")
            print(diff)

if __name__ == '__main__':
    wvs = word_vectors()
    wvs.train('data/dist_sim_data.txt')
    print("="*80)
    print("Comparing word vectors for \'dogs\' using [RAW_COUNT] and [PPMI]")
    wvs.get_word_vec('dogs', word_vectors.USE_RAW_COUNT, True)
    wvs.get_word_vec('dogs', word_vectors.USE_PPMI, True)
    print("-"*80)
    wvs.compute_distance("women", "men", word_vectors.USE_PPMI, True)
    wvs.compute_distance("women", "men", word_vectors.USE_REDUCED_PPMI, True)
    print("-"*80)
    wvs.compute_distance("women", "dogs", word_vectors.USE_PPMI, True)
    wvs.compute_distance("women", "dogs", word_vectors.USE_REDUCED_PPMI, True)
    print("-"*80)
    wvs.compute_distance("men", "dogs", word_vectors.USE_PPMI, True)
    wvs.compute_distance("men", "dogs", word_vectors.USE_REDUCED_PPMI, True)
    print("-"*80)
    wvs.compute_distance("feed", "like", word_vectors.USE_PPMI, True)
    wvs.compute_distance("feed", "like", word_vectors.USE_REDUCED_PPMI, True)
    print("-"*80)
    wvs.compute_distance("feed", "bite", word_vectors.USE_PPMI, True)
    wvs.compute_distance("feed", "bite", word_vectors.USE_REDUCED_PPMI, True)
    print("-"*80)
    wvs.compute_distance("like", "bite", word_vectors.USE_PPMI, True)
    wvs.compute_distance("like", "bite", word_vectors.USE_REDUCED_PPMI, True)
    print("="*80)
