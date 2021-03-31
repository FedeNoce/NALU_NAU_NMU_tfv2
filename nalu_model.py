import tensorflow as tf

import numpy as np
import time
import csv

import sys


class NALU_Model():
    def __init__(self, op,  p, w_star, hidden_size=2, input_dim=(0, 0), batch_size=128, num_epochs=1000, seed=3000,
                 epsilon=1e-8):
        self.p = p
        self.op = op
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.epsilon = epsilon
        self.hidden_size = hidden_size
        self.input_dim = input_dim
        self.weights_shape1 = (self.p, self.hidden_size)
        self.weights_shape2 = (self.hidden_size, 1)
        self.solved_at_iteration_step = 0

        self.W_hat1 = tf.Variable(tf.random.normal(shape=self.weights_shape1), name='W_hat1', trainable=True)
        self.M_hat1 = tf.Variable(tf.random.normal(shape=self.weights_shape1), name='M_hat1', trainable=True)
        self.G1 = tf.Variable(tf.random.normal(shape=self.weights_shape1), name='G1', trainable=True)
        self.W1 = tf.Variable(tf.random.normal(shape=self.weights_shape1), name='W1', trainable=False)
        self.W_hat2 = tf.Variable(tf.random.normal(shape=self.weights_shape2), name='W_hat2', trainable=True)
        self.M_hat2 = tf.Variable(tf.random.normal(shape=self.weights_shape2), name='M_hat2', trainable=True)
        self.G2 = tf.Variable(tf.random.normal(shape=self.weights_shape2), name='G2', trainable=True)
        self.W2 = tf.Variable(tf.random.normal(shape=self.weights_shape2), name='W2', trainable=False)
        self.iteration = 0
        self.w_star = w_star
        self.loss = tf.keras.losses.MeanSquaredError()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    def compute_logits(self, X):
        # First Layer NALU
        self.W1 = tf.nn.tanh(self.W_hat1) * tf.nn.sigmoid(self.M_hat1)
        a1 = tf.matmul(X, self.W1)
        g1 = tf.nn.sigmoid(tf.matmul(X, self.G1))
        m1 = tf.exp(tf.matmul(tf.math.log(tf.abs(X) + self.epsilon), self.W1))

        y1 = (g1 * a1) + (1 - g1) * m1
        # Second Layer NALU
        self.W2 = tf.nn.tanh(self.W_hat2) * tf.nn.sigmoid(self.M_hat2)
        a2 = tf.matmul(y1, self.W2)

        g2 = tf.nn.sigmoid(tf.matmul(y1, self.G2))
        m2 = tf.exp(tf.matmul(tf.math.log(tf.abs(y1) + self.epsilon), self.W2))

        y2 = (g2 * a2) + (1 - g2) * m2

        return y2

    def get_sparsity_error(self):
        W_abs = tf.math.abs(1 - tf.math.abs(self.W1))
        return np.array(tf.reduce_max(tf.math.minimum(W_abs, tf.math.abs(self.W1))))

    def forward(self, X, Y):
        with tf.GradientTape() as tape:
            f = self.compute_logits(X)
            loss = self.loss(f, Y)
        gradients = tape.gradient(loss, [self.W_hat1, self.M_hat1, self.G1, self.W_hat2, self.M_hat2, self.G2])
        self.optimizer.apply_gradients(
            zip(gradients, [self.W_hat1, self.M_hat1, self.G1, self.W_hat2, self.M_hat2, self.G2]))
        return loss

    def compute_loss(self, X, Y):
        f = self.compute_logits(X)
        loss = self.loss(f, Y)
        return loss

    def fit(self, x_train, y_train, x_val, y_val, x_test, y_test):
        W1_star = tf.constant(self.w_star)
        if self.op == 'sum':
            W2_star = tf.constant([[1., 1.]])
            y2 = tf.matmul(tf.matmul(x_test, tf.transpose(W1_star)), tf.transpose(W2_star))
            loss = self.loss(y2, y_test)
        if self.op == 'sub':
            W2_star = tf.constant([[1., -1.]])
            y2 = tf.matmul(tf.matmul(x_test, tf.transpose(W1_star)), tf.transpose(W2_star))
            loss = self.loss(y2, y_test)
        if self.op == 'mul':
            W2_star = tf.constant([[1., 1.]])
            y1 = tf.matmul(x_test, tf.transpose(W1_star))
            y2 = tf.reduce_prod(tf.math.multiply(y1, W2_star) + 1 - W2_star, 1)
            y2 = tf.expand_dims(y2, axis=1)
            loss = self.loss(y2, y_test)
        n, p = x_train.shape
        assert p == self.p
        for epoch in range(self.num_epochs):
            start_time = time.time()
            idx = list(range(n))
            np.random.shuffle(idx)
            losses = []
            n_batches = n // self.batch_size
            for b in range(n_batches):
                mb_idx = np.array(idx[b * self.batch_size:(b + 1) * self.batch_size])
                x_mb = x_train[mb_idx]
                y_mb = y_train[mb_idx]
                l = self.forward(x_mb, y_mb)
                self.iteration = self.iteration + 1
                losses.append(l)
            test_l = self.compute_loss(x_test, y_test)
            train_l = np.array(losses).mean()
            se = self.get_sparsity_error()
            if test_l > loss:
                self.solved_at_iteration_step = self.iteration
            if se < 10e-5:
                return epoch
                break
            if epoch % 1000 == 0:
                fmt = 'ep: {} t_l: {}  v_l: {} s_e:{} s:{:.4f} t: {}'
                print(fmt.format(epoch, train_l, test_l, se, (time.time() - start_time), loss))

    def predict(self, X_test):
        n, p = X_test.shape
        assert p == self.p
        pred = self._compute_logits(X_test)
        return pred



