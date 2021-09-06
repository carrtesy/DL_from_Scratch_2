import sys

import numpy as np
import matplotlib.pyplot as plt
from common.util import preprocess, create_co_matrix, cos_similarity, most_similar, ppmi

text = "You say goodbye and I say Hello."
corpus, word_to_id, id_to_word = preprocess(text)

print(corpus)
print(id_to_word)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)
print(cos_similarity(C[word_to_id["you"]], C[word_to_id["i"]]))
most_similar("you", word_to_id, id_to_word, C, top = 5)
W = ppmi(C)

np.set_printoptions(precision=3)
print(C)
print("-" * 50)
print("PPMI")
print(W)

U,S,V = np.linalg.svd(W)
print("U", U)
print("S", S)
print("V", V)

for word, word_id in word_to_id.items():
    plt.annotate(word, (U[word_id, 0], U[word_id, 1]))

plt.scatter(U[:,0], U[:,1], alpha=0.5)
plt.show()
