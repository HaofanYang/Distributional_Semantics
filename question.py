import numpy as np
import random
from collections import defaultdict

class synonym_question():
    # Create a list of synonym_question, as a numpy array
    @classmethod
    def create_questions_from_file(cls, root, num_of_questions):
        print("Loading synonym questions from [{s}]....".format(s=root))
        questions = np.empty(num_of_questions, dtype=synonym_question)
        for i in range(num_of_questions):
            with open("{r}/{index}.txt".format(r = root, index = i)) as f:
                lines = f.readlines()
                target_word = lines[0][:-1]
                answer = lines[1][:-1]
                non_synonyms = set(lines[2].split())
                questions[i] = synonym_question(target_word, answer, non_synonyms)
        return questions

    def __init__(self, word, synonym, non_synonyms):
        self.word = word  # string
        self.answer = synonym  # string
        self.non_synonyms = non_synonyms  # set
        self.options = [self.answer]
        for word in self.non_synonyms:
            self.options.append(word)
        random.shuffle(self.options)
        self.options = np.array(self.options)

    # Return a numpy array of options, including the answer and non-synonyms
    def get_options(self):
        return self.options

    # Return the word, of which we need to find the synonym, using word vectors
    def get_topic(self):
        return self.word

    # Return a vectorized lambda function
    # that return 1 if the given answer is the correct result. Return 0 otherwise
    def get_evaluator(self):
        return lambda predicted_word: 1 if predicted_word == self.answer else 0
    
    def __str__(self):
        topic = "Topic: {t}".format(t = self.word)
        options = "Options: {os}".format(os = self.get_options())
        answer = "Answer: {a}".format(a = self.answer)
        return "{a}\n{b}\n{c}\n-------".format(a = topic, b = options, c = answer)

class sat_question():
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
    def create_questions_from_file(cls, path):
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
    def get_options(self):
        return self.options

    # Return a tuple
    def get_topic(self):
        return self.topic

    # Return a vectorized lambda function
    # that return 1 if the given answer is the correct result. Return 0 otherwise
    def get_evaluator(self):
        return lambda predicted_answer_index: 1 if predicted_answer_index == self.answer else 0

    def __str__(self):
        topic = "Topic: {t}".format(t = self.topic)
        options = "Options: {os}".format(os = self.options)
        answer = "Answer: {a}".format(a = self.answer)
        return "{a}\n{b}\n{c}\n-------".format(a = topic, b = options, c = answer)
