# m - number of samples
# n - number of features 

# x (n, m) - training samples
# y (1, m) - training results
# w (n, 1) - parameters to be estimated in the logistic regression
# z (1, m) - intermediate sigmoid of wx + b
# a (1, m) - y hat
# dz = (1, m) - z derivative
#  

# in this example
# m = 3
# n = 4

import numpy as np
def sigmoid (x): return 1/(1 + np.exp(-x)) 

x = np.array([[1, 2, 3, 4],[2, 3, 4, 5], [3, 4, 5, 6]]).T
y = np.array([0, 0, 1]).reshape(1, 3)
b = 0
w = np.array([0, 0, 0, 0]).reshape(4,1)
dw = np.array([0, 0, 0, 0]).reshape(4,1)
db = 0
m = 3
n = 4


#single step gradient descent for logistic regression

alpha = 0.01

for i in range(50000):
    z = np.dot(w.T,x) + b
    a = sigmoid(z)
    dz = a - y
    dw = 1.0/float(m) * np.dot(x, dz.T)
    db = 1.0/float(m) * np.sum(dz)
    w = w - alpha * dw
    b = b - alpha * db
    print(dz)
    print(w)
    print(b)
