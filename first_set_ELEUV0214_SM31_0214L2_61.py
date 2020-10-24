import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import sklearn

# reading through the filtered set of data to train and test
energy_data = pd.read_excel('ELEUV0214_SM2_0214HMBA_21.xlsx')
df = pd.DataFrame(energy_data, columns=['Tagname', 'Timestamp', 'Value'])
# print(df)
# print(df.shape)


timestamp = []
value = []
for row in range(len(df)):
    timestamp.append(df["Timestamp"][row])
    value.append(df["Value"][row])

# print(timestamp)
# print(value)


# creating a visual graph of the data
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 15
fig_size[1] = 5
plt.rcParams["figure.figsize"] = fig_size
plt.plot(timestamp, value)
plt.title('Energy Consumption vs Time')
plt.ylabel('Energy Consumption')
plt.xlabel('Time')
plt.grid(True)
plt.autoscale(axis='x', tight=True)
#print(plt.plot(df["Value"]))
plt.show()


n_feature = 2
n_class = 2
n_iter = 10


def make_network(n_hidden=100):
    model = dict(
        W1=np.random.randn(n_feature, n_hidden),
        W2=np.random.randn(n_hidden, n_class)
    )
    return model

# print(make_network())


"""
FORWARD PROPAGATION
"""
def softmax(x):
    return np.exp(x) / np.exp(x).sum()


def forward(x, model):
    h = x @ model['W1']
    h[h < 0] = 0
    prob = softmax(h @ model['W2'])
    return h, prob

print(forward)


"""
BACKWARD PROPAGATION
"""
def backward(model, xs, hs, errs):
    dW2 = hs.T @ errs
    dh = errs @ model['W2'].T
    dh[hs <= 0] = 0
    dW1 = xs.T @ dh
    return dict(W1=dW1, W2=dW2)

print(backward)


def sgd(model, X_train, y_train, minibatch_size):
    for iter in range(n_iter):
        print('Iteration {}'.format(iter))

        X_train, y_train = shuffle(X_train, y_train)
        for i in range(0, X_train.shape[0], minibatch_size):
            X_train_mini = X_train[i:i + minibatch_size]
            y_train_mini = y_train[i:i + minibatch_size]
            model = sgd_step(model, X_train_mini, y_train_mini)
    return model


def sgd_step(model, X_train, y_train):
    grad = get_minibatch_grad(model, X_train, y_train)
    model = model.copy()
    for layer in grad:
        model[layer] += 1e-4 * grad[layer]
    return model


def get_minibatch_grad(model, X_train, y_train):
    xs, hs, errs = [], [], []
    for x, cls_idx in zip(X_train, y_train):
        h, y_pred = forward(x, model)
        y_true = np.zeros(n_class)
        y_true[int(cls_idx)] = 1.
        err = y_true - y_pred
        xs.append(x)
        hs.append(h)
        errs.append(err)
    return backward(model, np.array(xs), np.array(hs), np.array(errs))


minibatch_size = 50
n_experiment = 100

accs = np.zeros(n_experiment)

for k in range(n_experiment):
    model = make_network()
    model = sgd(model, X_train, y_train, minibatch_size)
    y_pred = np.zeros_like(y_test)
    for i, x in enumerate(X_test):
        _, prob = forward(x, model)
        y = np.argmax(prob)
        y_pred[i] = y
    accs[k] = (y_pred == y_test).sum() / y_test.size
print('Mean accuracy: {}, std: {}'.format(accs.mean(), accs.std()))