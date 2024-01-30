import torch 

X = torch.tensor([1,2,3,4])
Y = torch.tensor([2,4,6,8])



w = torch.tensor(0.00 , requires_grad=True)

def forward(x):
    return w * x

def loss(y,y_pred):
    return ((y_pred-y)**2).mean()


print(f'Prediction Before Training f(5) = {forward(5):.3f}')

learning_rate = 0.01
n_iters = 100

for epoch in range(n_iters):
       y_pred = forward(X)

       l = loss(Y,y_pred)

       l.backward()

       with torch.no_grad():  
            w-= learning_rate * w.grad

       w.grad.zero_()

       if(epoch % 10 == 0):
            print(f'epoch = {epoch + 1}, weight = {w:.3f}  loss = {l:.8f} ')

print(f'Prediction After Training f(5) = {forward(5)}')    