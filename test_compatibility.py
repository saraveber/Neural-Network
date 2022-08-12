
import numpy as np
from numpy import hamming
import pandas as pd
from sklearn.model_selection import train_test_split
import copy
from datetime import datetime


def mse(real, predictions):
    return np.mean(np.power(real - predictions, 2));

def mse_prime(real, predictions):
    return 2*(predictions-real)/real.size;

def mse_prime_numerical (a,b):
    eps = 1e-6
    d_num = []
    for i in range (10):
        b_add = copy.deepcopy(b)
        b_add[i]= b_add[i] + eps 
        b_sub = copy.deepcopy(b)
        b_sub[i] = b_sub[i] - eps 
        d_num.append((mse(a,b_add)-mse(a,b_sub))/(2*eps))
    return np.array(d_num)




def log_loss(real, predictions):
    eksp = np.log(predictions)
    eksp = eksp[np.arange(len(real)),real]
    final_cost = np.sum(eksp)
    return -final_cost/len(predictions)

def log_loss_prime(real, predictions):
    grad = np.zeros(predictions.shape)
    grad[np.arange(len(real)),real] = -1/(predictions[np.arange(len(real)),real]+1e-10)
    return grad/len(predictions)


def log_los_prime_numerical(b,a):
    eps = 1e-5
    d = []

    for i in range (10):
        d1 = []
        for j in range(2):
            a_add = copy.deepcopy(a)
            a_sub = copy.deepcopy(a)
            a_add[i,j] = a_add[i,j]+eps
            a_sub[i,j] = a_sub[i,j]-eps
            d1.append((log_loss(b, a_add)-log_loss(b,a_sub)) /(2*eps))
        d.append(d1)
    d = np.array(d)
    return d

a1 = np.ones((10,1))
a2 = np.random.rand(10,1)
a1 = a1-a2

a = np.hstack([a1,a2])

b = np.random.randint(0,2,(10,1)).flatten()



print(log_loss_prime(b,a))
print(log_los_prime_numerical(b,a))


if (log_loss_prime(b,a).all() == log_los_prime_numerical(b,a).all()):
	print("Je enako!")