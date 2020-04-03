import numpy as np
import matplotlib.pyplot as plt


def load_data1(n=20000):
    assert n % 2 == 0
    gaussian1 = np.random.normal(loc=-1, scale=0.25, size=(n//2,))
    gaussian2 = np.random.normal(loc=0.5, scale=0.5, size=(n//2,))
    data = (np.concatenate([gaussian1, gaussian2]) + 1).reshape([-1, 1])
    scaled_data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
    return 2 * scaled_data -1

def load_data2(n=20000):
    assert n % 2 == 0
    gaussian1 = np.random.normal(loc=-1, scale=0.03, size=(n//2,))
    gaussian2 = np.random.normal(loc=1, scale=0.03, size=(n//2,))
    data = (np.concatenate([gaussian1, gaussian2]) + 1).reshape([-1, 1])
    scaled_data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
    return 2 * scaled_data - 1

def load_warmup_data(n=20000):
    gaussian1 = np.random.normal(loc=-1, scale=0.25, size=(n//2,))
    data = gaussian1.reshape([-1, 1]) + 1
    scaled_data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
    return 2 * scaled_data -1

""" 
train_data = load_data1()
train_data2 = load_data2()
warmup_data = load_warmup_data()

plt.hist(train_data, bins=50)
plt.show()

plt.hist(train_data2, bins=200)
plt.show()

plt.hist(warmup_data, bins=200)
plt.show()
"""
