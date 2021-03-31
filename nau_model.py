import tensorflow as tf

import numpy as np
import time


class NAU_Model():
    def __init__(self,add , p, w_star, hidden_size=2, input_dim=(0, 0), batch_size=128, num_epochs=1000, seed=30060):
        self.p = p
        self.add = add
        self.lambda_start = 20000
        self.lambda_end = 30000
        self.lambda_sparse_hat = 0.1
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.hidden_size = hidden_size
        self.input_dim = input_dim
        self.w_star = w_star
        self.weights_shape1 = (self.hidden_size, self.p)
        self.weights_shape2 = (1, self.hidden_size)
        self.xavier_value_1 = tf.math.sqrt((6 / (self.p + self.hidden_size)))
        self.xavier_value_2 = tf.math.sqrt((6 / (self.hidden_size + 1)))
        self.iteration = 0
        self.solved_at_iteration_step = 0

        self.W1 = tf.Variable(
            tf.random.uniform(shape=self.weights_shape1, minval=-self.xavier_value_1, maxval=self.xavier_value_1),
            name='W1', constraint=lambda x: tf.clip_by_value(x, -1, 1))
        self.W2 = tf.Variable(
            tf.random.uniform(shape=self.weights_shape2, minval=-self.xavier_value_2, maxval=self.xavier_value_2),
            name='W2', constraint=lambda x: tf.clip_by_value(x, -1, 1))
        self.loss = tf.keras.losses.MeanSquaredError()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    def get_sparsity_error(self):
        W_abs = tf.math.abs(1 - tf.math.abs(self.W1))
        return tf.reduce_max(tf.math.minimum(W_abs, tf.math.abs(self.W1)))

    def get_lambda_sparse(self, t):
        return self.lambda_sparse_hat * tf.maximum(
            tf.minimum((t - self.lambda_start) / (self.lambda_end - self.lambda_start), 1), 0)

    def compute_logits(self, X):
        # First Layer NAU
        y1 = tf.matmul(X, tf.transpose(self.W1))
        # Second Layer NAU
        y2 = tf.matmul(y1, tf.transpose(self.W2))
        return y2

    def forward(self, X, Y, t):
        with tf.GradientTape() as tape:
            f = self.compute_logits(X)
            loss = self.loss(Y, f) + self.get_lambda_sparse(t) * self.get_regularizer(self.W1) + self.get_lambda_sparse(
                t) * self.get_regularizer(self.W2)
        gradients = tape.gradient(loss, [self.W1, self.W2])
        self.optimizer.apply_gradients(zip(gradients, [self.W1, self.W2]))
        return loss

    def compute_loss(self, X, Y):
        f = self.compute_logits(X)
        loss = self.loss(Y, f)
        return loss

    def fit(self, x_train, y_train, x_val, y_val, x_test, y_test):
        W1_star = tf.constant(self.w_star)
        if self.add == True:
            W2_star = tf.constant([[1., 1.]])
        else:
            W2_star = tf.constant([[1., -1.]])
        target = tf.matmul(tf.matmul(x_test, tf.transpose(W1_star)), tf.transpose(W2_star))
        loss = tf.reduce_mean(tf.losses.mean_squared_error(target, y_test))
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
                l = self.forward(x_mb, y_mb, epoch)
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

    def get_regularizer(self, w):
        m = w.shape[0]
        n = w.shape[1]
        sum = 0.
        const = 1 / (m * n)
        sum = tf.reduce_sum(tf.math.minimum(tf.math.abs(w), 1 - tf.math.abs(w)))  # Works
        return const * sum

