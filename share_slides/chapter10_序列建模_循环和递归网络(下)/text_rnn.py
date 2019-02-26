# encoding : utf-8
"""
@author: future
@email: weilai5@jd.com
"""
import tensorflow as tf
import numpy as np
import os
import datetime


class TextRNN:

    def __init__(self, vocab_size, num_class, seq_len, word_dim=128, hidden_dim=256):
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_class = num_class
        self.seq_len = seq_len

        self.build_graph()

    def build_graph(self):
        self.input_x = tf.placeholder(tf.int32, [None, self.seq_len], name="input_x")
        self.input_y = tf.placeholder(tf.int32, [None, ], name="input_y")
        gd = tf.one_hot(self.input_y, self.num_class)
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        with tf.name_scope('word_embedding'):
            embed_matrix = tf.Variable(tf.random_uniform(shape=(self.vocab_size + 1, self.word_dim)), name='word_embed')
            text_embed = tf.nn.embedding_lookup(embed_matrix, self.input_x, name='text_embed')

        with tf.name_scope('rnn'):
            forward_cell = tf.nn.rnn_cell.GRUCell(self.hidden_dim)
            backward_cell = tf.nn.rnn_cell.GRUCell(self.hidden_dim)
            contexts, final_states = tf.nn.bidirectional_dynamic_rnn(forward_cell, backward_cell, text_embed,
                                                                     dtype='float32')
            final_state = tf.concat(final_states, axis=1)
            print(final_state)

            final_state_drop = tf.nn.dropout(final_state, keep_prob=self.dropout_keep_prob)

        with tf.name_scope('fc'):
            pred_scores = tf.layers.dense(final_state_drop, self.num_class)

        self.pred = tf.argmax(pred_scores)

        with tf.name_scope('loss'):
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=gd, logits=pred_scores, name='ce_loss')
            self.loss = tf.reduce_mean(loss, name='loss')
            tf.summary.scalar('loss', self.loss)
        self.global_step = tf.Variable(1, name="global_step", trainable=False)


if __name__ == '__main__':
    np.random.seed(100)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    with tf.Session() as sess:
        rnn_model = TextRNN(vocab_size=5000, num_class=20, seq_len=30)

        train_op = tf.train.AdamOptimizer().minimize(rnn_model.loss, global_step=rnn_model.global_step)
        # opt = tf.train.AdamOptimizer()
        # gradients_vars = opt.compute_gradients(rnn_model.loss)
        # print(gradients_vars)
        # train_op = opt.apply_gradients(gradients_vars)

        timestamp = datetime.datetime.now().isoformat()[:-7]
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", 'example'))
        print("model path: {}\n".format(out_dir))

        # merge Summaries
        train_summary_op = tf.summary.merge_all()
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        data = np.random.randint(0, 5000, size=(5000, 30)).astype('int32')
        y = np.random.randint(0, 19, size=(5000,)).astype('int32')
        batch_size = 100
        batch_per_epoch = 5000 // 100
        print('trianing...')
        for k in range(50):
            for i in range(batch_per_epoch):
                feed = {
                    rnn_model.input_x: data[batch_size * i: batch_size * i + batch_size],
                    rnn_model.input_y: y[batch_size * i: batch_size * i + batch_size],
                    rnn_model.dropout_keep_prob: 0.5,
                }

                _, step, summaries, loss = sess.run(
                    [train_op, rnn_model.global_step, train_summary_op, rnn_model.loss], feed)
                if step % 10 == 0:
                    print('step', step, 'loss', loss)
                    train_summary_writer.add_summary(summaries, step)

                if step % 100 == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=step)
                    print("Saved model checkpoint to {}\n".format(path))
