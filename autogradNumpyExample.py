import numpy as np 

X = np.array([1,2,3,4])
Y = np.array([2,4,6,8])

w = 0.00

def forward(x):
    return w * x

def loss(y,y_pred):
    return ((y_pred-y)**2).mean()

def gradient(x,y,y_pred):
    return np.dot(2*x,y_pred-y).mean()

print(f'Prediction Before Training f(5) = {forward(5):.3f}')

learning_rate = 0.01
n_iters = 10

for epoch in range(n_iters):
       y_pred = forward(X)

       l = loss(Y,y_pred)

       dw = gradient(X,Y,y_pred)

       w-= learning_rate * dw

       if(epoch % 1 == 0):
            print(f'epoch = {epoch + 1}, weight = {w:.3f}  loss = {l:.8f} ')

print(f'Prediction After Training f(5) = {forward(5)}')    