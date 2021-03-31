import tensorflow as tf
import numpy as np
import time
import csv
from nau_model import NAU_Model
from nmu_model import NMU_Model
from nalu_model import NALU_Model


def dataset_build_sum(input_size, subset_ratio, overlap_ratio, range, k1, k2):
    x = np.random.uniform(range[0], range[1], input_size)
    a = np.sum(x[:, int(k1 * input_size[1]): int(input_size[1] * (subset_ratio + k1))], axis=1)
    b = np.sum(x[:, int(k2 * input_size[1]): int(input_size[1] * (subset_ratio + k2))], axis=1)
    t = a + b
    t = np.expand_dims(t, axis=1)
    return x.astype('float32'), t.astype('float32')


def dataset_build_prod(input_size, subset_ratio, overlap_ratio, range, k1, k2):
    x = np.random.uniform(range[0], range[1], input_size)
    a = np.sum(x[:, int(k1 * input_size[1]): int(input_size[1] * (subset_ratio + k1))], axis=1)
    b = np.sum(x[:, int(k2 * input_size[1]): int(input_size[1] * (subset_ratio + k2))], axis=1)
    t = np.multiply(a, b)
    t = np.expand_dims(t, axis=1)
    return x.astype('float32'), t.astype('float32')


def dataset_build_sub(input_size, subset_ratio, overlap_ratio, range, k1, k2):
    x = np.random.uniform(range[0], range[1], input_size)
    a = np.sum(x[:, int(k1 * input_size[1]): int(input_size[1] * (subset_ratio + k1))], axis=1)
    b = np.sum(x[:, int(k2 * input_size[1]): int(input_size[1] * (subset_ratio + k2))], axis=1)
    t = a - b
    t = np.expand_dims(t, axis=1)
    return x.astype('float32'), t.astype('float32')


def build_w_star(sample_size, subset_ratio, k1, k2):
    w_star = np.zeros([2, 100]) + 10e-5
    w_star[0, int(k1 * sample_size): int(sample_size * (subset_ratio + k1))] = 1 - 10e-5
    w_star[1, int(k2 * sample_size): int(sample_size * (subset_ratio + k2))] = 1 - 10e-5
    return w_star.astype('float32')

#DOT
k2 = np.random.uniform(0, 1-(0.25))
k1 = np.random.uniform(0, 1-(0.25))
w_star = build_w_star(100, 0.25, k1, k2)
x_train, y_train = dataset_build_prod([1000, 100], 0.25, 0.5, [1, 2], k1, k2)
x_val, y_val = dataset_build_prod([1000, 100], 0.25, 0.5, [1, 2], k1, k2)
x_test, y_test = dataset_build_prod([1000, 100], 0.25, 0.5, [2, 6], k1, k2)

print('Start Nalu Models trainings for multiplication')
for i in range(10):
    Nalu = NALU_Model(op='mul', p=x_train.shape[1], w_star=w_star, hidden_size=2, input_dim=x_train.shape,
                      batch_size=64, num_epochs=50000, seed=1000 + i)
    epoch = Nalu.fit(x_train, y_train, x_val, y_val, x_test, y_test)
    k2 = np.random.uniform(0, 1 - (0.25))
    k1 = np.random.uniform(0, 1 - (0.25))
    w_star = build_w_star(100, 0.25, k1, k2)
    x_train, y_train = dataset_build_prod([1000, 100], 0.25, 0.5, [1, 2], k1, k2)
    x_val, y_val = dataset_build_prod([1000, 100], 0.25, 0.5, [1, 2], k1, k2)
    x_test, y_test = dataset_build_prod([1000, 100], 0.25, 0.5, [2, 6], k1, k2)
    with open('/home/fn7045399/new_data/nalu_prod.csv', 'a') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow([str(i), epoch, Nalu.get_sparsity_error(), Nalu.solved_at_iteration_step])


print('Start NMU Models trainings')
for i in range(10):
  Nmu = NMU_Model(p=x_train.shape[1], w_star=w_star, hidden_size=2, input_dim=x_train.shape,batch_size=64,num_epochs=50000, seed=1000+i)
  epoch = Nmu.fit(x_train,y_train,x_val,y_val,x_test, y_test)
  k2 = np.random.uniform(0, 1 - (0.25))
  k1 = np.random.uniform(0, 1 - (0.25))
  w_star = build_w_star(100, 0.25, k1, k2)
  x_train, y_train = dataset_build_prod([1000,100], 0.25, 0.5, [1,2], k1, k2)
  x_val, y_val = dataset_build_prod([1000, 100], 0.25, 0.5, [1, 2], k1, k2)
  x_test, y_test = dataset_build_prod([1000, 100], 0.25, 0.5, [2, 6], k1, k2)
  with open('/home/fn7045399/new_data/nmu_prod.csv', 'a') as file:
      csv_writer = csv.writer(file)
      csv_writer.writerow([str(i), epoch, Nmu.get_sparsity_error(), Nmu.solved_at_iteration_step])

#SUM
k2 = np.random.uniform(0, 1-(0.25))
k1 = np.random.uniform(0, 1-(0.25))
w_star = build_w_star(100, 0.25, k1, k2)
x_train, y_train = dataset_build_sum([1000, 100], 0.25, 0.5, [1, 2], k1, k2)
x_val, y_val = dataset_build_sum([1000, 100], 0.25, 0.5, [1, 2], k1, k2)
x_test, y_test = dataset_build_sum([1000, 100], 0.25, 0.5, [2, 6], k1, k2)

print('Start Nalu Models trainings for addition')
for i in range(10):
    Nalu = NALU_Model(op='sum', p=x_train.shape[1], w_star=w_star, hidden_size=2, input_dim=x_train.shape,
                      batch_size=64, num_epochs=50000, seed=1000 + i)
    epoch = Nalu.fit(x_train, y_train, x_val, y_val, x_test, y_test)
    k2 = np.random.uniform(0, 1 - (0.25))
    k1 = np.random.uniform(0, 1 - (0.25))
    w_star = build_w_star(100, 0.25, k1, k2)
    x_train, y_train = dataset_build_sum([1000, 100], 0.25, 0.5, [1, 2], k1, k2)
    x_val, y_val = dataset_build_sum([1000, 100], 0.25, 0.5, [1, 2], k1, k2)
    x_test, y_test = dataset_build_sum([1000, 100], 0.25, 0.5, [2, 6], k1, k2)
    with open('/home/fn7045399/new_data/nalu_sum.csv', 'a') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow([str(i), epoch, Nalu.get_sparsity_error(), Nalu.solved_at_iteration_step])

print('Start Nau Models trainings for addition')
for i in range(10):
  Nau = NAU_Model(add=True,p=x_train.shape[1], w_star=w_star, hidden_size=2, input_dim=x_train.shape,batch_size=64,num_epochs=50000, seed=1000+i)
  epoch = Nau.fit(x_train,y_train,x_val,y_val,x_test, y_test)
  k2 = np.random.uniform(0, 1 - (0.25))
  k1 = np.random.uniform(0, 1 - (0.25))
  w_star = build_w_star(100, 0.25, k1, k2)
  x_train, y_train = dataset_build_sum([1000,100], 0.25, 0.5, [1,2], k1, k2)
  x_val, y_val = dataset_build_sum([1000, 100], 0.25, 0.5, [1, 2], k1, k2)
  x_test, y_test = dataset_build_sum([1000, 100], 0.25, 0.5, [2, 6], k1, k2)
  with open('/home/fn7045399/new_data/nau_sum.csv', 'a') as file:
      csv_writer = csv.writer(file)
      csv_writer.writerow([str(i), epoch, Nau.get_sparsity_error(), Nau.solved_at_iteration_step])


#SUB
k2 = np.random.uniform(0, 1-(0.25))
k1 = np.random.uniform(0, 1-(0.25))
w_star = build_w_star(100, 0.25, k1, k2)
x_train, y_train = dataset_build_sub([1000, 100], 0.25, 0.5, [1, 2], k1, k2)
x_val, y_val = dataset_build_sub([1000, 100], 0.25, 0.5, [1, 2], k1, k2)
x_test, y_test = dataset_build_sub([1000, 100], 0.25, 0.5, [2, 6], k1, k2)

print('Start Nalu Models trainings for Subtraction')
for i in range(10):
    Nalu = NALU_Model(op='sub', p=x_train.shape[1], w_star=w_star, hidden_size=2, input_dim=x_train.shape,
                      batch_size=64, num_epochs=50000, seed=1000 + i)
    epoch = Nalu.fit(x_train, y_train, x_val, y_val, x_test, y_test)
    k2 = np.random.uniform(0, 1 - (0.25))
    k1 = np.random.uniform(0, 1 - (0.25))
    w_star = build_w_star(100, 0.25, k1, k2)
    x_train, y_train = dataset_build_sub([1000, 100], 0.25, 0.5, [1, 2], k1, k2)
    x_val, y_val = dataset_build_sub([1000, 100], 0.25, 0.5, [1, 2], k1, k2)
    x_test, y_test = dataset_build_sub([1000, 100], 0.25, 0.5, [2, 6], k1, k2)
    with open('/home/fn7045399/new_data/nalu_sub.csv', 'a') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow([str(i), epoch, Nalu.get_sparsity_error(), Nalu.solved_at_iteration_step])

print('Start Nau Models trainings for subtraction')
for i in range(10):
  Nau = NAU_Model(add=False, p=x_train.shape[1], w_star=w_star, hidden_size=2, input_dim=x_train.shape,batch_size=64,num_epochs=50000, seed=1000+i)
  epoch = Nau.fit(x_train, y_train, x_val, y_val, x_test, y_test)
  k2 = np.random.uniform(0, 1 - (0.25))
  k1 = np.random.uniform(0, 1 - (0.25))
  w_star = build_w_star(100, 0.25, k1, k2)
  x_train, y_train = dataset_build_sub([1000,100], 0.25, 0.5, [1,2], k1, k2)
  x_val, y_val = dataset_build_sub([1000, 100], 0.25, 0.5, [1, 2], k1, k2)
  x_test, y_test = dataset_build_sub([1000, 100], 0.25, 0.5, [2, 6], k1, k2)
  with open('/home/fn7045399/new_data/nau_sub.csv', 'a') as file:
      csv_writer = csv.writer(file)
      csv_writer.writerow([str(i), epoch, Nau.get_sparsity_error(), Nau.solved_at_iteration_step])
