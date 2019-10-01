from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import os
import numpy as np

class MySentences(object):
    def __init__(self, filePath):
        self.filePath = filePath

    def __iter__(self):
        with open(self.filePath, mode="r" ,encoding='utf-8') as fr:
            lines = fr.readlines()
            for line in lines:
                line = line.strip().split('_')
                yield line

def train():
    sentence = MySentences('./data/corpus.txt')

    model = Word2Vec(min_count=5,
                     window=5,
                     size=300,
                     workers=4)
    model.build_vocab(sentences=sentence)
    model.train(sentences=sentence, epochs=10, total_examples=1671403)

    model.wv.save('word2vec.wv')

def load_and_save(vocab_size, embedding_dim):
    wv = KeyedVectors.load("./data/word2vec.wv", mmap='r')
    keys = wv.vocab

    embedding_mat = np.random.normal(-0.25, 0.25, (vocab_size, embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    for key in keys:
        try:
            embedding_mat[int(key), :] = wv[key]
        except:
            print("error", key)
    embedding_mat[21225, :] = 0.0

    np.save(file='./data/word2vec.npy', arr=embedding_mat)

if __name__=='__main__':
    # train()
    load_and_save(21226, 300)
