import numpy as np
import os,sys
from dataset.mnist import load_mnist
from PIL import Image
import pickle
def step_function(x):
    # 阶跃函数
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    # sigmoid 函数
    return 1/(1+np.exp(-x))


def relu(x):
    # ReLU 函数
    return np.maximum(0, x)


def identity_function(x):
    # 恒等函数
    return x


def softmax(a):
    # softmax 函数
    c=np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a/sum_exp_a
    return y


""" def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])
    return network """


def forword(network, x):
    # 前向函数
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1)+b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2)+b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3)+b3
    y = softmax(a3)
    return y
def get_data():
    #训练图像，训练标签，测试图像，测试标签
    (x_train,t_train),(x_test,t_test)=load_mnist(flatten=True,normalize=True,one_hot_label=False)
    return x_test,t_test
def init_network():
    with open('sample_weight.pkl','rb') as f:
        network=pickle.load(f)
    return network

""" x,t=get_data()
network=init_network()
accurency_cnt=0
for i in range(len(x)):
    y=forword(network,x[i])
    p=np.argmax(y)
    if p==t[i]:
        accurency_cnt+=1 """
x, t = get_data()
network = init_network()
batch_size = 100  # 批处理数量
accurency_cnt = 0
for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = forword(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accurency_cnt += np.sum(p == t[i:i+batch_size])
print(f'Accurency:{float(accurency_cnt/len(x))}')


