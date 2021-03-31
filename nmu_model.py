
import tensorflow as tf

import numpy as np
import time
import csv

import sys
class NMU_Model():
    def __init__(self, p, w_star, hidden_size=2, input_dim=(0, 0), batch_size=128, num_epochs=1000, seed=3900):
        self.p = p
        self.lambda_start_nmu = 20000
        self.lambda_end_nmu = 40000
        self.lambda_sparse_hat_nmu = 100
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.hidden_size = hidden_size
        self.input_dim = input_dim
        self.w_star = w_star
        self.weights_shape1 = (self.hidden_size, self.p)
        self.weights_shape2 = (1, self.hidden_size)
        self.xavier_value_1 = tf.math.sqrt((6 / (self.p + self.hidden_size)))
        self.iteration = 0

        self.W1 = tf.Variable(
            tf.random.uniform(shape=self.weights_shape1, minval=-self.xavier_value_1, maxval=self.xavier_value_1,
                              dtype=tf.float32), name='W1', constraint=lambda x: tf.clip_by_value(x, -1, 1))
        self.W2 = tf.Variable(tf.random.normal(shape=self.weights_shape2, mean=0.5, stddev=0.25, dtype=tf.float32),
                              name='W2', constraint=lambda x: tf.clip_by_value(x, 0, 1))

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.loss = tf.keras.losses.MeanSquaredError()

    def get_sparsity_error(self):
        with tf.device('/device:GPU:0'):
            W_abs = tf.math.abs(1 - tf.math.abs(self.W1))
            return tf.reduce_max(tf.math.minimum(W_abs, tf.math.abs(self.W1)))

    def get_lambda_sparse(self, t):
        with tf.device('/device:GPU:0'):
            return self.lambda_sparse_hat_nmu * tf.maximum(
                tf.minimum(((t - self.lambda_start_nmu) / (self.lambda_end_nmu - self.lambda_start_nmu)), 1), 0)

    def _compute_logits(self, X):
        with tf.device('/device:GPU:0'):
            # First Layer NAU
            y1 = tf.matmul(X, tf.transpose(self.W1))
            # Second Layer NMU
            y2 = tf.reduce_prod(y1 * self.W2 + 1 - self.W2, 1)
            y2 = tf.expand_dims(y2, axis=1)
            return y2

    def _step(self, X, Y, t):
        with tf.GradientTape() as tape:
            f = self._compute_logits(X)
            loss = self.loss(f, Y) + self.get_lambda_sparse(t) * self.get_nau_regularizer(
                self.W1) + self.get_lambda_sparse(t) * self.get_nmu_regularizer(self.W2)
        gradients = tape.gradient(loss, [self.W1, self.W2])
        self.optimizer.apply_gradients(zip(gradients, [self.W1, self.W2]))
        return loss

    def _metrics(self, X, Y, t):
        f = self._compute_logits(X)
        loss = self.loss(f, Y)
        return loss

    def fit(self, x_train, y_train, x_val, y_val, x_test, y_test):
        W1_star = tf.constant(self.w_star)
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
                l = self._step(x_mb, y_mb, epoch)
                self.iteration = self.iteration + 1
                losses.append(l)
            test_l = self._metrics(x_test, y_test, self.iteration)
            train_l = np.array(losses).mean()
            if test_l > loss:
                self.solved_at_iteration_step = self.iteration
            se = self.get_sparsity_error()
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

    def get_nau_regularizer(self, w):
        with tf.device('/device:GPU:0'):
            m = w.shape[0]
            n = w.shape[1]
            sum = 0.
            const = 1 / (m * n)
            sum = tf.reduce_sum(tf.math.minimum(tf.math.abs(w), 1 - tf.math.abs(w)))  # Works
            return const * sum

    def get_nmu_regularizer(self, w):
        with tf.device('/device:GPU:0'):
            m = w.shape[0]
            n = w.shape[1]
            sum = 0.
            const = 1 / (m * n)
            sum = tf.reduce_sum(tf.math.minimum(w, 1 - w))  # Works
            return const * sum


