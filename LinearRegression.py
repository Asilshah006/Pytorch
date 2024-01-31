import numpy as np
import torch
import torch.nn as nn
from sklearn import datasets
import matplotlib.pyplot as plt


X_numpy , y_numpy = datasets.make_regression(n_samples=100 , n_features=1 , noise=20 , random_state=1)

X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0] , 1)

n_samples, n_features  = X.shape

inputSize = n_features
outputSize = 1

model = nn.Linear(inputSize , outputSize)

learning_rate = 0.01
n_iters = 100

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters() , lr=learning_rate)

for epoch in range(n_iters):
    y_pred = model(X)

    loss = criterion(y_pred , y)

    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    if(epoch + 1) % 10 == 0:
        print(f'epoch = {epoch} loss = {loss.item():.4f}')

predicted = model(X).detach()

plt.plot(X_numpy , y_numpy , 'ro')
plt.plot(X_numpy , predicted , 'b')
plt.show()




