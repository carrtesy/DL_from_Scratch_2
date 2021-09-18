import numpy as np
from common.layers import MatMul, SoftmaxWithLoss
from common.util import preprocess, create_contexts_target, convert_one_hot

c = np.array([[1, 0, 0, 0, 0, 0, 0]])
W = np.random.randn(7, 3)
layer = MatMul(W)
h = layer.forward(c)
print(h)

c0 = np.array([[1, 0, 0, 0, 0, 0, 0]])
c1 = np.array([[0, 0, 1, 0, 0, 0, 0]])

W_in = np.random.randn(7, 3)
W_out = np.random.randn(3, 7)

in_layer0 = MatMul(W_in)
in_layer1 = MatMul(W_in)
out_layer = MatMul(W_out)

h0 = in_layer0.forward(c0)
h1 = in_layer1.forward(c1)

h = 0.5 * (h0 + h1)
s = out_layer.forward(h)

print(s)

text = "You say goodbye and I say hello."
corpus, word_to_id, id_to_word = preprocess(text)

print(corpus)
print(id_to_word)

contexts, target = create_contexts_target(corpus, window_size = 1)

print(contexts)
print(target)

vocab_size = len(word_to_id)
target = convert_one_hot(target, vocab_size)
contexts = convert_one_hot(contexts, vocab_size)



class SimpleCBOW:
    def __init__(self, vocab_size, hidden_size):
        V, H = vocab_size, hidden_size

        W_in = 0.01 * np.random.randn(V, H).astype("f")
        W_out = 0.01 * np.random.randn(H, V).astype("f")

        self.in_layer0 = MatMul(W_in)
        self.in_layer1 = MatMul(W_in)
        self.out_layer = MatMul(W_out)
        self.loss_layer = SoftmaxWithLoss()

        layers = [self.in_layer0, self.in_layer1, self.out_layer]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        self.word_vecs = W_in

        def forward(self, contexts, target):
            h0 = self.in_layer0.forward(contexts[:, 0])
            h1 = self.in_layer1.forward(contexts[:, 1])
            h = 0.5 * (h0 + h1)
            score = self.out_layer.forward(h)
            loss = self.loss_layer.forward(score, target)
            return loss

        def backward(self, dout = 1):
            ds = self.loss_layer.backward(dout)
            da = self.out_layer.backward(ds)
            da *= 0.5
            self.in_layer1.backward(da)
            self.in_layer2.backward(da)
            return None