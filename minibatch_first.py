import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import check_array


# reading through the filtered set of data to train and test
energy_data = pd.read_excel('ELEUV0214_SM2_0214HMBA_21.xlsx')
df = pd.DataFrame(energy_data, columns=['Tagname', 'Timestamp', 'Value'])
#print(df)
# print(df.shape)

#timestamp = []
#value = []
#for row in range(len(df)):
#    timestamp.append(df["Timestamp"][row])
#    value.append(df["Value"][row])

#print(timestamp)
#print(value)

x_train, x_test, Y_train, Y_test = train_test_split(df["Timestamp"], df["Value"], test_size=0.2, random_state=0)
print(x_train)
#print(Y_train)
# creating predictions
def hypothesis(x, theta):
    return np.dot(x, theta)

# gradient error
def gradient(x, Y, theta):
    hyp = hypothesis(x, theta)
    grad = np.dot(x.transpose(), (hyp - Y))
    return grad

# get new dot product
def cost(x, Y, theta):
    hyp = hypothesis(x, theta)
    new_dot = np.dot((hyp - Y).transpose(), (hyp - Y))
    new_dot = new_dot / 2
    return new_dot[0]

# make minibatches
def make_mini_batch(x, Y, size):
    mini_batches = []
    data = np.hstack((x, Y))
    np.random.shuffle(data)
    n_minibatches = data.shape[0] // size
    for i in range(n_minibatches + 1):
        mini_batch = data[i * size:(i + 1) * size, :]
        x_mini = mini_batch[:, :-1]
        Y_mini = mini_batch[:, -1].reshape((-1, 1))
        mini_batches.append((x_mini, Y_mini))
    if data.shape[0] % size != 0:
        mini_batch = data[i * size:data.shape[0]]
        x_mini = mini_batch[:, :-1]
        Y_mini = mini_batch[:, -1].reshape((-1, 1))
        mini_batches.append((x_mini, Y_mini))
    return mini_batches


# calculating gradient descent
def gradientDescent(x, Y, learning_rate=0.001, size=100):
    theta = np.zeros((x.shape[1], 1))
    error_list = []
    max_iters = 3
    for itr in range(max_iters):
        mini_batches = make_mini_batch(x, Y, size)
        for mini_batch in mini_batches:
            x_mini, Y_mini = mini_batch
            theta = theta - learning_rate * gradient(x_mini, Y_mini, theta)
            error_list.append(cost(x_mini, Y_mini, theta))
    return theta, error_list

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = check_array(y_true, y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

theta, error_list = gradientDescent(x_train, Y_train)
print("Bias = ", theta[0])
print("Coefficients = ", theta[1: ])

plt.plot(error_list)
plt.xlabel("Number of iterations")
plt.ylabel("Cost")
plt.show()

# predicting output for X_test
y_pred = hypothesis(x_test, theta)
plt.scatter(x_test[:, 1], Y_test[:, ], marker = '.')
plt.plot(x_test[:, 1], y_pred, color = 'orange')
plt.show()

# calculating error in predictions
print(mean_absolute_percentage_error(Y_test, y_pred))

