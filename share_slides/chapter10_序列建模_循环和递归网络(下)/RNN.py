# encoding: utf-8
import numpy as np

# 一个句子10个词，词向量的维度是128
# RNN隐含状态的维度是 64
seq_len = 10
word_dim = 128
dim_hidden = 64
lr = 1e-3

sentences = np.random.uniform(size=(10, 128)).astype('float32')
W_h = np.random.uniform(size=(128, 64)).astype('float32')
W_x = np.random.uniform(size=(64, 64)).astype('float32')


def RNNCell(h, x):
    return np.tanh(np.dot(h, W_h) + np.dot(x, W_x))


def RNN(seqs, reverse=False):
    """
    :param seqs: the input sequence
    :param reverse: for bi-directional RNN
    :return:
            output: the final hidden state
            all_hidden: all hidden states
    """
    t_step = seqs.shape[0]
    all_hiddens = []
    hidden_t1 = np.random.uniform(size=(1, dim_hidden))

    recurrent = range(t_step - 1, -1, -1) if reverse else range(t_step)
    for i in recurrent:
        hidden_t1 = RNNCell(hidden_t1, seqs[i, :])
        all_hiddens.append(hidden_t1)
    if reverse:
        all_hiddens = all_hiddens[::-1]
    return all_hiddens[-1], all_hiddens


def compute_gradients(loss, W1, W2):
    W_h_grad = np.random.uniform(size=(128, 64)).astype('float32')
    W_x_grad = np.random.uniform(size=(64, 64)).astype('float32')

    for i in range(seq_len):
        pass
    return W_h_grad, W_x_grad


def update()