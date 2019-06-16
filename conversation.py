import os
import pickle
import nlu
import random
import numpy as np
from tqdm.auto import tqdm
from sklearn.neighbors import KDTree


def get_vec(text):
    v = nlu.SPACY_NLP(text).vector
    v = v / sum(v**2)**0.5
    return v


class SimpleConversation:
    def __init__(self, filename='gc_knn.pkl', k=5):
        self.filename = filename
        self.k = k
        if filename is not None and os.path.exists(filename):
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.knn = data['knn']
                self.phrases = data['phrases']
        else:
            self.index = None
            self.phrases = []

    def reply(self, text):
        v = get_vec(text)
        indices = self.knn.query(v.reshape(1, -1), k=self.k, return_distance=False)[0]
        index = random.choice(indices)
        return self.phrases[index][1]

    def train(self, pairs):
        self.phrases = pairs
        matrix = np.stack([
            get_vec(pair[0])
            for pair in tqdm(pairs)
        ])
        self.knn = KDTree(matrix, leaf_size=40)
        if self.filename is not None:
            with open(self.filename, 'wb') as f:
                pickle.dump({'knn': self.knn, 'phrases': self.phrases}, f)
