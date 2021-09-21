from functools import reduce
import numpy as np
import zipfile
import gensim
from scipy.spatial import distance
from gensim.models import KeyedVectors

# Load vectors directly from the file
from src.data_reader import DataReader
import gensim.downloader

class BiasedLexRank:

    def __init__(self, t):
        reader = DataReader()
        self.aspect_emb_s = reader.get_json(f'abae/aspect_emb_sentences_t{t}.json')
        self.topics = reader.get_json(f'abae/emb_aspects_50_t1.json')
        # model_file = 'models/182.zip'
        # with zipfile.ZipFile(model_file, 'r') as archive:
        # stream = archive.open('model.bin')
        # self.word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(stream, binary=True)
        # self.word2vec_model = KeyedVectors.load_word2vec_format('abae/word_emb', binary=False)
        #self.word2vec_model = KeyedVectors.load_word2vec_format('abae/word_emb_norm', binary=False)
        self.word2vec_model = gensim.downloader.load('word2vec-ruscorpora-300')

    def __call__(self, sentences, topic_description, d, sentences_count):
        # self.topic_description = self._get_mean_vector(topic_description[0])
        self.topic_index = topic_description
        self.topic_description = self.topics[topic_description]
        self.d = d
        self.init_sentences = sentences
        self.baseline_vector = self.baseline_ranking()
        self.sentences = [self._get_aspect_vector(s.strip()) for s in sentences]

        self.matrix = self.build_similarity_matrix()
        lex_rank_scores = self.do_lex_rank(10e-3)
        return self.get_sentences_ids(lex_rank_scores, sentences_count)

    def baseline_ranking(self):
        bias_nodes = []
        for sentence in self.init_sentences:
            sentence = sentence.strip()
            if sentence in self.aspect_emb_s:
                dict_sentence = self.aspect_emb_s[sentence]['r_s']
                if len(sentence) < 3:
                    bias_nodes.append(1)
                else:
                    s = self.aspect_emb_s[sentence]
                    #bias = 1 - self.aspect_emb_s[sentence]['probs'][0][self.topic_index]
                    bias = distance.cosine(self.topic_description, dict_sentence)
                    bias_nodes.append(bias)
            else:
                bias_nodes.append(1)
        bias_vec = np.array(bias_nodes)
        bias_sum = np.sum(bias_vec)

        if bias_sum == 0:
            bias_sum = 1

        bias_vec = bias_vec / bias_sum
        return bias_vec

    def _get_mean_vector(self, words):
        words = [word.strip() for word in words.split() if word.strip() in self.word2vec_model.vocab]
        if len(words) >= 1:
            return np.mean(self.word2vec_model[words], axis=0)
        else:
            return []

    def _get_aspect_vector(self, words):
        # words = [word.strip() for word in words.split() if word.strip() in self.word2vec_model.vocab]
        # if len(words) >= 1:
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
        sentences_count = len(self.sentences)
        matrix = np.zeros((sentences_count, sentences_count))
        for i in range(sentences_count):
            for j in range(i, sentences_count):
                if i != j:
                    if len(self.sentences[i]) == 0 or len(self.sentences[j]) == 0:
                    #if len(self.sentences[i]) < 3 or len(self.sentences[j]) < 3:
                        val = 1
                    else:
                        val = distance.cosine(self.sentences[i], self.sentences[j])
                    matrix[i][j] = val
                    matrix[j][i] = val

        row_sums = matrix.sum(axis=1, keepdims=True)

        matrix = matrix / row_sums
        res = self.d * self.baseline_vector + (1 - self.d) * matrix
        return res


    def do_lex_rank(self, epsilon):
        sentences_count = len(self.sentences)
        probabilities = np.ones(sentences_count) / sentences_count
        diff = 1
        iters = 0
        while diff > epsilon and iters < 1000:
            tmp = np.dot(self.matrix.T, probabilities)
            diff = np.linalg.norm(np.subtract(tmp, probabilities))
            probabilities = tmp
            iters += 1
        return probabilities


    @staticmethod
    def get_sentences_ids(lex_rank_scores, sentences_count):
        #a = sorted(enumerate(lex_rank_scores), key=lambda x: x[1])
        sorted_ids = [i[0] for i in sorted(enumerate(lex_rank_scores), key=lambda x: x[1])]
        return sorted_ids[:sentences_count]
