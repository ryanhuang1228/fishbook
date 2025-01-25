import torchvision
import numpy as np
from pathlib import Path

DIR = Path(__file__).resolve().parent

def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1
        
    return T

def load_mnist(one_hot_lable):
    train_data = torchvision.datasets.MNIST(
        root=DIR, train=True, transform=torchvision.transforms.ToTensor(), download=True
    )
    test_data = torchvision.datasets.MNIST(
        root=DIR, train=False, transform=torchvision.transforms.ToTensor(), download=True
    )

    # 数据预处理，flatten
    x_train = np.zeros((len(train_data), 28**2))
    t_train = np.zeros((len(train_data)), dtype=np.int32)
    for i in range(len(train_data)):
        x_train[i, :] = train_data[i][0].reshape(1,-1)
        t_train[i] = train_data[i][1]

    x_test = np.zeros((len(test_data), 28**2))
    t_test = np.zeros((len(test_data)), dtype=np.int32)
    for i in range(len(test_data)):
        x_test[i, :] = test_data[i][0].reshape(1,-1)
        t_test[i] = test_data[i][1]
    
    if one_hot_lable:
        t_train = _change_one_hot_label(t_train)
        t_test = _change_one_hot_label(t_test)
    return (x_train, t_train), (x_test, t_test)

def load_mnist_origin(one_hot_lable):
    train_data = torchvision.datasets.MNIST(
        root=DIR, train=True, transform=torchvision.transforms.ToTensor(), download=True
    )
    test_data = torchvision.datasets.MNIST(
        root=DIR, train=False, transform=torchvision.transforms.ToTensor(), download=True
    )

    # 数据预处理，flatten
    x_train = np.zeros((len(train_data), 1, 28, 28))
    t_train = np.zeros((len(train_data)), dtype=np.int32)
    for i in range(len(train_data)):
        x_train[i, :] = train_data[i][0]
        t_train[i] = train_data[i][1]

    x_test = np.zeros((len(test_data), 1, 28, 28))
    t_test = np.zeros((len(test_data)), dtype=np.int32)
    for i in range(len(test_data)):
        x_test[i, :] = test_data[i][0]
        t_test[i] = test_data[i][1]
    
    if one_hot_lable:
        t_train = _change_one_hot_label(t_train)
        t_test = _change_one_hot_label(t_test)
    return (x_train, t_train), (x_test, t_test)