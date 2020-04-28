import numpy as np
from collections import defaultdict
import random


class word_vectors:
    COSINE = "COSINE"
    NEGATIVE_EUCLIDEAN_DISTANCE = "NEGATIVE_EUCLIDEAN_DISTANCE"
    SIMILARITY_SWITCHER = {
        COSINE: lambda vec1, vec2: np.dot(vec1, vec2.T)
        / np.linalg.norm(vec1)
        / np.linalg.norm(vec2),
        NEGATIVE_EUCLIDEAN_DISTANCE: lambda vec1, vec2: 0 - np.linalg.norm(vec1 - vec2),
    }

    def __init__(self):
        self.matrix = None
        self.word_dict = defaultdict(int)
        self.vector_length = 0

    # Load trained word vectors from file.
    def load(self, path):
        print("Reading word vectors from [{s}]....".format(s = path))
        with open(path) as f:
            for line in f:
                line = line.split()
                if self.vector_length == 0:
                    self.vector_length = len(line) - 1
                word = line[0]
                word_index = self.word_dict.get(
                    word, len(self.word_dict)
                )  # Find the word index from dict. If not found, the next index will be the size of current dict
                self.word_dict[word] = word_index
        self.matrix = np.zeros((len(self.word_dict), self.vector_length))
        with open(path) as f:
            for line in f:
                line = line.split()
                word = line[0]
                word_index = self.word_dict[word]  # Find the word index from dict. If not found, the next index will be the size of current dict
                self.matrix[word_index] = np.array(line[1:])

    # Given a criteria (either cosine or distance) to compute similarity
    # Return a lambda function that takes a question as the input and return the predicted synonym
    def get_synonym_predictor(self, criteria):
        def predictor(synonym_question):
            options = synonym_question.show_options()
            word = synonym_question.show_target_word()
            f = lambda word2: self._similarity(word, word2, criteria)
            f = np.vectorize(f)
            similarities = f(options)  # Similarities between each option and the target word
            arg_max = similarities.argmax()
            return options[arg_max]
        return np.vectorize(predictor)
    
    # Given a criteria (either cosine or distance) to compute similarity
    # Return a lambda function that takes a sat question as the input and return the predicted answer index
    def get_sat_predictor(self, criteria):
        def predictor(sat_question):
            options = sat_question.show_options()
            topic = sat_question.show_topic()
            f = lambda pair2: self._pair_similarity(topic, pair2, criteria)
            f = np.vectorize(f)
            similarities = f(options)  # Similarities between each option and the target word
            arg_max = similarities.argmax()
            return arg_max
        return np.vectorize(predictor)

    # return the word vector of a word
    def _get_word_vector(self, word):
        # Return an unit vector when given UNK
        if not word in self.word_dict:
            to_return = np.zeros((1, self.vector_length))
            return to_return - np.sqrt(1 / self.vector_length)
        word_index = self.word_dict[word]
        return self.matrix[word_index]

    # return the similarity of two words
    def _similarity(self, word1, word2, criteria):
        word_vec_1 = self._get_word_vector(word1)
        word_vec_2 = self._get_word_vector(word2)
        return word_vectors.SIMILARITY_SWITCHER[criteria](word_vec_1, word_vec_2)

    # return the similarity of two pairs of words
    def _pair_similarity(self, pair1, pair2, criteria):
        pair1_diff = self._get_word_vector(pair1[1]) - self._get_word_vector(pair1[0])
        pair21_plus_diff = self._get_word_vector(pair2[0]) + pair1_diff # this is the expected vector of pair2[1]
        return word_vectors.SIMILARITY_SWITCHER[criteria](pair21_plus_diff, self._get_word_vector(pair2[1]))


class synonym_question:
    # Create a list of synonym_question, as a numpy array
    @classmethod
    def create_questions_from_file(cls, path, num_of_questions):
        print("Creating synonym questions from [{s}]....".format(s=path))
        word_set = set()
        synonyms_of_words = defaultdict(set)  # synonyms of each word
        with open(path) as f:
            f.readline()  # Skip the first line
            for line in f:
                word1, word2 = line.lower().split()
                word_set.add(word1)
                word_set.add(word2)
                dict_of_word_1 = synonyms_of_words.get(word1, set())
                dict_of_word_1.add(word2)
                dict_of_word_2 = synonyms_of_words.get(word1, set())
                dict_of_word_2.add(word1)
                synonyms_of_words[word1] = dict_of_word_1
                synonyms_of_words[word2] = dict_of_word_2
        questions = np.empty(num_of_questions, dtype=synonym_question)
        for i in range(num_of_questions):
            target_word = random.sample(word_set, 1)[0]
            synonyms = synonyms_of_words[target_word]
            answer = random.sample(synonyms, 1)[0]
            non_synonyms = word_set.difference(synonyms)
            non_synonyms = set(random.sample(non_synonyms, 4))
            questions[i] = synonym_question(target_word, answer, non_synonyms)
        return questions

    def __init__(self, word, synonym, non_synonyms):
        self.word = word  # string
        self.answer = synonym  # string
        self.non_synonyms = non_synonyms  # set

    # Return a numpy array of options, including the answer and non-synonyms
    def show_options(self):
        options = [self.answer]
        for word in self.non_synonyms:
            options.append(word)
        return np.array(options)

    # Return the word, of which we need to find the synonym, using word vectors
    def show_target_word(self):
        return self.word

    # Return a vectorized lambda function
    # that return 1 if the given answer is the correct result. Return 0 otherwise
    def get_evaluator(self):
        return lambda predicted_word: 1 if predicted_word == self.answer else 0


class sat_question:
    # mapping alphabet answers to index
    ALPHAPET_TO_INDEX = {
        'a': 0,
        'b': 1,
        'c': 2,
        'd': 3,
        'e': 4,
    }

    # Create a list of synonym_question, as a numpy array
    @classmethod
    def create_questions_from_file(cls, path, num_of_questions):
        print("Creating SAT questions from [{s}]....".format(s=path))
        ignore_sign = '#'
        questions = []
        with open(path) as f:
            lines = f.readlines()
            pointer = 0
            while pointer < len(lines):
                line = lines[pointer]
                # Skip all empty lines and comments
                if line[0] == ignore_sign or line == '\n': 
                    pointer += 1 
                    continue
                # Skip the heading
                pointer += 1
                topic = tuple(lines[pointer].split()[:-1])
                pointer += 1
                options = np.empty(5, dtype = tuple)
                for i in range(5):
                    option = tuple(lines[pointer].split()[:-1])
                    options[i] = option
                    pointer += 1
                alphabet_answer = lines[pointer].strip()
                pointer += 1
                index_of_alphabet = sat_question.ALPHAPET_TO_INDEX[alphabet_answer]
                questions.append(sat_question(topic, index_of_alphabet, options))
        return questions
                

    def __init__(self, _topic, _answer, _options):
        self.topic = _topic  # tuple, the topic of the question
        self.answer = _answer  # int, the index of correct answer in options
        self.options = _options  # a numpy array of tuples, containing available options (1 correct and 4 wrong)

    # Return a numpy array of tuples
    def show_options(self):
        return self.options

    # Return a tuple
    def show_topic(self):
        return self.topic

    # Return a vectorized lambda function
    # that return 1 if the given answer is the correct result. Return 0 otherwise
    def get_evaluator(self):
        return lambda predicted_answer_index: 1 if predicted_answer_index == self.answer else 0

def synonym_detections(wvs, questions, num_of_questions):
    # Vectorized lambda function that takes questions as input and return answers of each questions
    cosine_predictors = wvs.get_synonym_predictor(word_vectors.COSINE)
    distance_predictors = wvs.get_synonym_predictor(
        word_vectors.NEGATIVE_EUCLIDEAN_DISTANCE
    )
    # Apply vectorized lambda function on questions
    predicted_synonyms_cosine = cosine_predictors(questions)
    predicted_synonyms_distance = distance_predictors(questions)
    # An array of lambda functions
    evaluators = np.array([question.get_evaluator() for question in questions])
    apply_evaluators = np.vectorize(lambda f, x: f(x))
    # Compute an array for each question, where 0 denotes wrong answer and 1 denotes correct answer
    evaluation_cosine = apply_evaluators(evaluators, predicted_synonyms_cosine)
    evaluation_distance = apply_evaluators(evaluators, predicted_synonyms_distance)
    # Sum over evaluation arrays, we can get the number of correct answers, from which we can further compute accuracies
    cosine_accuracy = evaluation_cosine.sum() / len(evaluation_cosine)
    distance_accuracy = evaluation_distance.sum() / len(evaluation_distance)
    print("COSINE accuracy: " + str(cosine_accuracy))
    print("DISTANCE accuracy: " + str(distance_accuracy))

def sat_detections(wvs, questions, num_of_questions):
    # Vectorized lambda function that takes questions as input and return answers of each questions
    cosine_predictors = wvs.get_sat_predictor(word_vectors.COSINE)
    distance_predictors = wvs.get_sat_predictor(
        word_vectors.NEGATIVE_EUCLIDEAN_DISTANCE
    )
    # Apply vectorized lambda function on questions
    predicted_sat_cosine = cosine_predictors(questions)
    predicted_sat_distance = distance_predictors(questions)
    # An array of lambda functions
    evaluators = np.array([question.get_evaluator() for question in questions])
    apply_evaluators = np.vectorize(lambda f, x: f(x))
    # Compute an array for each question, where 0 denotes wrong answer and 1 denotes correct answer
    evaluation_cosine = apply_evaluators(evaluators, predicted_sat_cosine)
    evaluation_distance = apply_evaluators(evaluators, predicted_sat_distance)
    # Sum over evaluation arrays, we can get the number of correct answers, from which we can further compute accuracies
    cosine_accuracy = evaluation_cosine.sum() / len(evaluation_cosine)
    distance_accuracy = evaluation_distance.sum() / len(evaluation_distance)
    print("COSINE accuracy: " + str(cosine_accuracy))
    print("DISTANCE accuracy: " + str(distance_accuracy))

if __name__ == "__main__":
    synonym_questions = synonym_question.create_questions_from_file("data/EN_syn_verb.txt", 1000)
    wvs_word2vec = word_vectors()
    wvs_word2vec.load("data/GoogleNews-vectors-rcv_vocab.txt")
    synonym_detections(wvs_word2vec, synonym_questions, 1000)
    wvs_compose = word_vectors()
    wvs_compose.load("data/EN-wform.w.2.ppmi.svd.500.rcv_vocab.txt")
    synonym_detections(wvs_compose, synonym_questions, 1000)

    sat_questions = sat_question.create_questions_from_file("data/SAT-package-V3.txt", 1000)
    sat_detections(wvs_word2vec, sat_questions, 1000)
    sat_detections(wvs_compose, sat_questions, 1000)
