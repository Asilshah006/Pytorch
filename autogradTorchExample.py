import torch 
import torch.nn as nn

X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
Y = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)

X_test = torch.tensor([5] , dtype=torch.float32)

# X = torch.tensor([1,2,3,4], dtype=torch.float32)
# Y = torch.tensor([2,4,6,8], dtype=torch.float32)

n_samples , n_features = X.shape

input_size = n_features
output_size = n_features

model = nn.Linear(input_size,output_size)

# def forward(x):
#     return w * x

# def loss(y,y_pred):
#     return ((y_pred-y)**2).mean()


print(f'Prediction Before Training f(5) = {model(X_test).item():.3f}')

learning_rate = 0.02
n_iters = 100

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_iters):
       y_pred = model(X)

       l = loss(Y,y_pred)

       l.backward()

       optimizer.step()

       optimizer.zero_grad()

       if(epoch % 10 == 0):
            [w , b] = model.parameters()
            print(f'epoch = {epoch + 1}, weight = {w[0][0]:.3f}  loss = {l:.8f} ')

print(f'Prediction After Training f(5) = {model(X_test).item():.3f}')    