import torch
import numpy as np

x = torch.rand(2,2)
a = torch.ones(1,3)
b = torch.zeros(1,3)
c = torch.tensor([2.5,0.1,0.3])
d = torch.empty(2,3, dtype=torch.float16)
y = torch.rand(2,2)

z = x + y
z = torch.add(x,y)


# e = torch.rand(4,4)
# print(e.view(16))
# print(e.view(-1,2))


f = torch.ones(5)
print(f)
g = f.numpy()
f.add_(1)
print(g)


# print(z)


# y.add_(x)
# print(x[1,1].item())
# print(d.dtype)
# print(d.size())

# print(c)
# print(a)
# print(b)
# print(x)