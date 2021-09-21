from functools import reduce
import numpy as np
import zipfile
import gensim
from scipy.spatial import distance
from gensim.models import KeyedVectors

# Load vectors directly from the file
from src.data_reader import DataReader
import gensim.downloader


class AbaeLexRank:

    def __init__(self, t=1, multiple=1):
        reader = DataReader()
        self.aspect_emb_s = reader.get_json(f'abae/aspect_emb_sentences_t{t}.json')
        self.topics = reader.get_json(f'abae/emb_aspects_50_t{t}.json')
        self.word2vec_model = KeyedVectors.load_word2vec_format('abae/word_emb_norm', binary=False)
        self.multiple = multiple

    def __call__(self, sentences, topic_description, d, sentences_count):
        self.topic_num = topic_description
        self.topic_description = self.topics[topic_description]
        self.d = d
        self.init_sentences = sentences
        self.sentences = [self._get_aspect_vector(s.strip()) for s in sentences]

        self.matrix = self.build_similarity_matrix()
        lex_rank_scores = self.do_lex_rank(10e-3)
        return self.get_sentences_ids(lex_rank_scores, sentences_count, len(self.sentences), self.topic_description)

    def _get_mean_vector(self, words):
        words = [word.strip() for word in words.split() if word.strip() in self.word2vec_model.vocab]
        if len(words) >= 1:
            return np.mean(self.word2vec_model[words], axis=0)
        else:
            return []

    def _get_aspect_vector(self, words):
        if words in self.aspect_emb_s:
            return self.aspect_emb_s[words]['r_s']
        else:
            return []


    @staticmethod
    def _get_gen_sentence_probability(self, sentence_u, sentence_v):
        p = 1
        for word in sentence_u:
            p *= self._get_gen_word_probability(word, sentence_v)
        return p ** (1 / len(sentence_u))


    @staticmethod
    def _get_gen_word_probability(word, sentence, smooth_coef=0.5):
        # TODO: smoothing
        p_in_sentence = sentence.count(word) / len(sentence)
        return 1e-10 if p_in_sentence == 0 else p_in_sentence

    def build_similarity_matrix(self):
        self.nodes = self.sentences.copy()
        #self.nodes.extend(self.topics) # add all topic vectors to the graph
        #topics_count = len(self.topics)

        self.nodes.extend([self.topic_description]) # add single aspect vector to the graph
        topics_count = 1

        sentences_count = len(self.sentences)
        all_count = sentences_count + topics_count
        matrix = np.zeros((all_count, all_count))

        for i in range(all_count):
            for j in range(i, all_count):
                if i != j:
                    if len(self.nodes[i]) == 0 or len(self.nodes[j]) == 0:
                        val = 1
                    else:
                        val = distance.cosine(self.nodes[i], self.nodes[j])
                    matrix[i][j] = val
                    matrix[j][i] = val

        row_sums = matrix.sum(axis=1, keepdims=True)

        matrix = matrix / row_sums

        # an emphasis on target aspect
        ind = topics_count + sentences_count - 1
        for i in range(all_count):
            if i != ind:
                 matrix[i][ind] *= self.multiple
                 matrix[ind][i] *= self.multiple
        matrix[ind][ind] *= self.multiple

        return matrix

    def do_lex_rank(self, epsilon):
        sentences_count = len(self.sentences)
        #topics_count = len(self.topics) # all topic vectors in the graph
        topics_count = 1 # single aspect vector in the graph

        all_count = sentences_count + topics_count
        probabilities = np.ones(all_count) / all_count
        diff = 1
        iters = 0
        while diff > epsilon and iters < 1000:
            tmp = np.dot(self.matrix.T, probabilities)
            diff = np.linalg.norm(np.subtract(tmp, probabilities))
            probabilities = tmp
            iters += 1
        return probabilities


    @staticmethod
    def get_sentences_ids(lex_rank_scores, sentences_count, size_sentences, topic_description):
        aspects_scores = sorted(enumerate(lex_rank_scores[size_sentences:]), key=lambda x: x[1])
        main_aspect = [i[0] for i in aspects_scores][0]

        sentences_scores = sorted(enumerate(lex_rank_scores[:size_sentences]), key=lambda x: x[1])
        sorted_ids = [i[0] for i in sentences_scores]
        return sorted_ids[:sentences_count], main_aspect == topic_description
