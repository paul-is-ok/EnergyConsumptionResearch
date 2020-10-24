import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

energy_data = pd.read_excel('ELEUV0214_SM2_0214HMBA_21.xlsx')
df = pd.DataFrame(energy_data, columns=['Tagname', 'Timestamp', 'Value'])

x_train, x_test, Y_train, Y_test = train_test_split(df["Timestamp"], df["Value"], test_size=0.2, random_state=0)

def hypothesis(x, theta):
    return np.dot(x, theta)


def gradient(x, Y, theta):
    h = hypothesis(x, theta)
    grad = np.dot(x.transpose(), (h-Y))
    return grad


def cost(x, Y, theta):
    h = hypothesis(x, theta)
    J = np.dot((h-Y).transpose(), (h - Y))
    J /= 2
    return J[0]


def create_mini_batches(x, Y, batch_size):
    mini_batches = []
    data = np.hstack((x, Y))
    np.random.shuffle(data)
    n_minibatches = data.shape[0] // batch_size
    for i in range(n_minibatches + 1):
        mini_batch = data[i * batch_size:(i + 1) * batch_size, :]
        X_mini = mini_batch[:, :-1]
        Y_mini = mini_batch[:, -1].reshape((-1, 1))
        mini_batches.append((X_mini, Y_mini))
    if data.shape[0] % batch_size != 0:
        mini_batch = data[i * batch_size:data.shape[0]]
        X_mini = mini_batch[:, :-1]
        Y_mini = mini_batch[:, -1].reshape((-1, 1))
        mini_batches.append((X_mini, Y_mini))
    return mini_batches


def gradientDescent(x, Y, learning_rate=0.001, batch_size=32):
    theta = np.zeros((x.shape[1], 1))
    error_list = []
    max_iters = 3
    for itr in range(max_iters):
        mini_batches = create_mini_batches(x, Y, batch_size)
        for mini_batch in mini_batches:
            x_mini, Y_mini = mini_batch
            theta = theta - learning_rate * gradient(x_mini, Y_mini, theta)
            error_list.append(cost(x_mini, Y_mini, theta))

    return theta, error_list


theta, error_list = gradientDescent(x_train, Y_train)
print("Bias = ", theta[0])
print("Coefficients = ", theta[1:])

# visualising gradient descent
plt.plot(error_list)
plt.xlabel("Number of iterations")
plt.ylabel("Cost")
plt.show()