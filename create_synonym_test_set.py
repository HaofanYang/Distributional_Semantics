from collections import defaultdict
import random
if __name__ == "__main__":
    path = "data/EN_syn_verb.txt"
    num_of_questions = 1000
    print("Creating synonym questions from [{s}]....".format(s=path))
    word_set = set()
    synonyms_of_words = defaultdict(set)  # synonyms of each word
    with open(path) as f:
        f.readline()  # Skip the first line
        for line in f:
            word1, word2 = line.lower().split()
            if (word1 == '0' or word2 =='0'): continue
            word_set.add(word1)
            word_set.add(word2)
            dict_of_word_1 = synonyms_of_words.get(word1, set())
            dict_of_word_1.add(word2)
            dict_of_word_2 = synonyms_of_words.get(word1, set())
            dict_of_word_2.add(word1)
            synonyms_of_words[word1] = dict_of_word_1
            synonyms_of_words[word2] = dict_of_word_2
    for i in range(num_of_questions):
        target_word = random.sample(word_set, 1)[0]
        synonyms = synonyms_of_words[target_word]
        answer = random.sample(synonyms, 1)[0]
        non_synonyms = word_set.difference(synonyms)
        if target_word in non_synonyms:
            non_synonyms.remove(target_word)
        non_synonyms = set(random.sample(non_synonyms, 4))
        name = "synonym_test_set_2/{index}.txt".format(index = i)
        with open(name, "w+") as f2:
            f2.write(target_word + "\n")
            f2.write(answer + "\n")
            for word in non_synonyms:
                f2.write(word + " ")