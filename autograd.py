import torch

# x = torch.rand(3, requires_grad=True)

# print(x)

# y = x + 2

# print(y)

# z = y*y*2
# z = z.mean()

# print(z)

# z.backward()

# print(x.grad)

# Removing Grad

# x.requires_grad_(False)
# a = x.detach()
# with torch.no_grad():
#     y = x + 2
#     print(y)

# print(a)


# Example

weights = torch.ones(4, requires_grad=True)

# for epoch in range(2):
#     model_output = (weights * 3).sum() 

#     model_output.backward()
#     print(weights.grad)

#     weights.grad.zero_() 


# Optim Method

optimizer = torch.optim.SGD(weights , lr=0.01)
optimizer.step()
optimizer.zero_grad()