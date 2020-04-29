import numpy as np
from collections import defaultdict
import random
from question import sat_question, synonym_question
import scipy.linalg as scipy_linalg

class word_vectors:
    COSINE = "COSINE"
    NEGATIVE_EUCLIDEAN_DISTANCE = "NEGATIVE_EUCLIDEAN_DISTANCE"
    SIMILARITY_SWITCHER = {
        COSINE: lambda vec1, vec2: np.dot(vec1, vec2.T) / scipy_linalg.norm(vec1) / scipy_linalg.norm(vec2),
        NEGATIVE_EUCLIDEAN_DISTANCE: lambda vec1, vec2: 0 - scipy_linalg.norm(vec1 - vec2),
    }

    def __init__(self, _name):
        self.matrix = None
        self.word_dict = defaultdict(int)
        self.vector_length = 0
        self.name = _name

    # Load trained word vectors from file.
    def load(self, path):
        print("MODEL [{n}]: Reading word vectors from \'{s}\'....".format(n = self.name, s = path))
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

    # Given a list of questions, evaluate the performance
    def predict_and_evaluate(self, questions):
        # Vectorized lambda function that takes questions as input and return answers of each questions
        if isinstance(questions[0], synonym_question):
            print("Predicting and evaluating MODEL [{s}] on Synonym questions...".format(s = self.name))
            cosine_predictors = self.get_synonym_predictor(word_vectors.COSINE)
            distance_predictors = self.get_synonym_predictor(
                word_vectors.NEGATIVE_EUCLIDEAN_DISTANCE
            )
        else:
            print("Predicting and evaluating MODEL [{s}] on SAT questions...".format(s = self.name))
            cosine_predictors = self.get_sat_predictor(word_vectors.COSINE)
            distance_predictors = self.get_sat_predictor(
                word_vectors.NEGATIVE_EUCLIDEAN_DISTANCE
            )
        # Apply vectorized lambda function on questions
        cosine_predicted_results = cosine_predictors(questions)
        distance_predicted_results = distance_predictors(questions)
        # An array of lambda functions
        evaluators = np.array([question.get_evaluator() for question in questions])
        apply_evaluators = np.vectorize(lambda f, x: f(x))
        # Compute an array for each question, where 0 denotes wrong answer and 1 denotes correct answer
        evaluation_cosine = apply_evaluators(evaluators, cosine_predicted_results)
        evaluation_distance = apply_evaluators(evaluators, distance_predicted_results)
        # Sum over evaluation arrays, we can get the number of correct answers, from which we can further compute accuracies
        cosine_accuracy = evaluation_cosine.sum() / len(evaluation_cosine)
        distance_accuracy = evaluation_distance.sum() / len(evaluation_distance)
        print("COSINE accuracy: " + str(cosine_accuracy))
        print("DISTANCE accuracy: " + str(distance_accuracy))
        print("-"*80)

    # Given a criteria (either cosine or distance) to compute similarity
    # Return a lambda function that takes a question as the input and return the predicted synonym
    def get_synonym_predictor(self, criteria):
        def predictor(synonym_question):
            options = synonym_question.get_options()
            word = synonym_question.get_topic()
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
            options = sat_question.get_options()
            topic = sat_question.get_topic()
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
            return to_return + np.sqrt(1 / self.vector_length)
        word_index = self.word_dict[word]
        return self.matrix[word_index]

    # return the similarity of two words
    def _similarity(self, word1, word2, criteria):
        word_vec_1 = np.zeros((1, self.vector_length))
        for word in word1.split("_"):
            if word == 'to': continue
            word_vec_1 += self._get_word_vector(word)
        word_vec_2 = np.zeros((1, self.vector_length))
        for word in word2.split("_"):
            if word == 'to': continue
            word_vec_2 += self._get_word_vector(word)
        return word_vectors.SIMILARITY_SWITCHER[criteria](word_vec_1, word_vec_2)

    # return the similarity of two pairs of words
    def _pair_similarity(self, pair1, pair2, criteria):
        pair1_diff = self._get_word_vector(pair1[1]) - self._get_word_vector(pair1[0])
        pair21_plus_diff = self._get_word_vector(pair2[0]) + pair1_diff # this is the expected vector of pair2[1]
        return word_vectors.SIMILARITY_SWITCHER[criteria](pair21_plus_diff, self._get_word_vector(pair2[1]))

if __name__ == "__main__":
    # Loading models
    print("="*80)
    wvs_word2vec = word_vectors("word2vec")
    wvs_word2vec.load("data/GoogleNews-vectors-rcv_vocab.txt")
    wvs_compose = word_vectors("compose")
    wvs_compose.load("data/EN-wform.w.2.ppmi.svd.500.rcv_vocab.txt")
    # Creating testing questins
    sat_questions = sat_question.create_questions_from_file("data/SAT-package-V3.txt")
    synonym_questions = synonym_question.create_questions_from_file("synonym_test_set", 1000)
    print("-"*80)
    # Predict and evaluate
    wvs_word2vec.predict_and_evaluate(synonym_questions)
    wvs_compose.predict_and_evaluate(synonym_questions)
    wvs_word2vec.predict_and_evaluate(sat_questions)
    wvs_compose.predict_and_evaluate(sat_questions)
    print("="*80)
